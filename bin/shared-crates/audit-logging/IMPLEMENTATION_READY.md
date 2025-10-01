# Audit Logging - Implementation Ready

**Status**: All blockers resolved ✅  
**Date**: 2025-10-01  
**Ready for**: Source code implementation

---

## Blockers Resolved

### 1. ✅ Dependencies Added

**Cargo.toml** now includes:

```toml
[features]
default = []
platform = ["hmac", "ed25519-dalek", "reqwest"]

[dependencies]
# Core dependencies
thiserror.workspace = true
serde = { workspace = true }
serde_json.workspace = true
tracing.workspace = true
chrono = { workspace = true }
tokio = { workspace = true }
futures = { workspace = true }

# Cryptography for hash chains
sha2 = "0.10"
hex = "0.4"

# Input validation (CRITICAL for security)
input-validation = { path = "../input-validation" }

# Optional platform mode
hmac = { version = "0.12", optional = true }
ed25519-dalek = { version = "2.0", optional = true }
reqwest = { version = "0.11", optional = true, features = ["json"] }
```

### 2. ✅ Input Validation Integration

**`src/validation.rs`** now uses `input-validation` crate:

```rust
fn sanitize(input: &str) -> Result<String> {
    input_validation::sanitize_string(input)
        .map_err(|e| AuditError::InvalidInput(e.to_string()))
}
```

**Security Impact**: Prevents log injection attacks (ANSI escapes, control chars, null bytes, Unicode overrides).

### 3. ✅ Cryptography Implementation

**`src/crypto.rs`** now has:

- ✅ `compute_event_hash()` — SHA-256 hash computation (FIPS 140-2 approved)
- ✅ `verify_hash_chain()` — Hash chain integrity verification
- ⏳ Platform mode signatures (HMAC/Ed25519) — Stubs remain for future implementation

**Hash includes**:
- audit_id
- timestamp (RFC3339)
- service_id
- event (JSON serialized)
- prev_hash

### 4. ✅ Path Validation

**`src/config.rs`** now has:

```rust
pub fn validate_audit_dir(path: &Path) -> Result<PathBuf>
```

**Security checks**:
- Path must be absolute
- Path must exist and be a directory
- Path must be within `/var/lib/llorch/audit`
- Resolves symlinks (prevents symlink attacks)

### 5. ✅ Configuration Fixed

- Fixed `Duration` import issue (changed to `flush_interval_secs: u64`)
- Added `platform` feature flag
- All `#[cfg(feature = "platform")]` warnings resolved

### 6. ✅ Compilation Verified

Both crates compile successfully:

```bash
✅ cargo check -p audit-logging
✅ cargo check -p audit-logging-bdd
```

Only warnings are for unused code (expected at this stage).

---

## What's Ready for Implementation

### Core Modules (Ready to Implement)

1. **`src/logger.rs`** — Main audit logger
   - `AuditLogger::new()` — Create channel, spawn writer task
   - `emit()` — Send events to channel
   - `flush()` — Flush buffered events
   - `shutdown()` — Graceful cleanup

2. **`src/writer.rs`** — Background writer task
   - Event batching (flush every 1 second or 100 events)
   - File writing with fsync
   - Hash computation and chain linking
   - Rotation trigger logic

3. **`src/storage.rs`** — File I/O
   - Append-only file writing
   - Manifest tracking
   - File rotation
   - Checksum computation

4. **`src/query.rs`** — Query and verification
   - File reading and parsing
   - Filtering (actor, time range, event type)
   - Pagination
   - Hash chain verification

### Security Features (Implemented)

✅ **Input validation** — `input-validation` crate integrated  
✅ **Hash chains** — SHA-256 computation and verification  
✅ **Path validation** — Symlink protection, directory traversal prevention  
✅ **Error types** — All security error variants defined  

### Testing Infrastructure (Ready)

✅ **BDD test suite** — 25+ scenarios for validation  
✅ **Unit tests** — 3 tests in `validation.rs` (more needed)  
⏳ **Integration tests** — Need to create `tests/` directory  

---

## Implementation Order (Recommended)

### Phase 1: Core Functionality (Week 1)

1. **Logger & Writer** (2-3 days)
   - Implement `AuditLogger::new()`
   - Implement background writer task
   - Implement `emit()` with channel
   - Add unit tests

2. **Storage** (2-3 days)
   - Implement file writing
   - Implement manifest tracking
   - Implement rotation
   - Add unit tests

### Phase 2: Query & Verification (Week 2)

3. **Query** (2-3 days)
   - Implement file reading
   - Implement filtering
   - Implement pagination
   - Add unit tests

4. **Verification** (1-2 days)
   - Integrate `verify_hash_chain()`
   - Implement checksum verification
   - Add integration tests

### Phase 3: Hardening (Week 3)

5. **Integration Tests** (2-3 days)
   - End-to-end audit flow
   - Tampering detection
   - Performance tests

6. **Documentation** (1-2 days)
   - Module-level docs
   - Usage examples
   - Security warnings

---

## What's NOT Blocking

These can be implemented later:

- ⏳ Platform mode (HMAC/Ed25519 signatures)
- ⏳ Query API rate limiting
- ⏳ Retention policy enforcement
- ⏳ SIEM export
- ⏳ Real-time streaming

---

## Security Checklist

Before production deployment:

- [ ] All user input sanitized via `input-validation`
- [ ] Hash chains verified on startup
- [ ] File permissions set correctly (700 for directory, 400 for files)
- [ ] Symlink protection enabled
- [ ] Path traversal prevention tested
- [ ] Buffer limits enforced (1000 events or 10MB)
- [ ] Graceful degradation on buffer full
- [ ] No secrets logged (use fingerprints)
- [ ] Audit log access is audited

---

## Dependencies Summary

**Total new dependencies**: 3 crates

- `sha2` — SHA-256 hashing (FIPS 140-2 approved)
- `hex` — Hex encoding for hashes
- `input-validation` — Log injection prevention (local path dependency)

**Optional dependencies** (platform mode):
- `hmac` — HMAC-SHA256 signatures
- `ed25519-dalek` — Ed25519 signatures
- `reqwest` — HTTP client for platform API

---

## Next Steps

1. **Start implementing `logger.rs`** — Core entry point
2. **Implement `writer.rs`** — Background task
3. **Add integration tests** — End-to-end scenarios
4. **Run BDD tests** — Verify validation works

---

**Status**: 🟢 Ready for implementation

All critical blockers resolved. Crate compiles successfully. BDD tests ready to verify validation logic.
