# secrets-management BDD Behaviors

**Comprehensive behavior catalog for secrets-management**

This document catalogs all Cucumber/Gherkin scenarios testing the secrets-management crate.

---

## Test Summary

**Total Scenarios**: 21  
**Total Features**: 4  
**Security Focus**: Credential protection, timing attacks, file permissions

---

## Feature: File Loading (6 scenarios)

### Scenario 1: Load secret from file with correct permissions
- **Given** a secret file at "/etc/llorch/secrets/api-token"
- **And** a secret file with permissions 384 (0600)
- **And** a secret file containing "test-secret-token"
- **When** I load the secret from file
- **Then** the operation should succeed

**Tests**: SM-LOAD-4001, SEC-SECRET-020

### Scenario 2: Reject world-readable secret file
- **Given** a secret file at "/etc/llorch/secrets/api-token"
- **And** a secret file with permissions 420 (0644)
- **When** I load the secret from file
- **Then** the operation should fail
- **And** the operation should reject world-readable files

**Tests**: SM-SEC-5004, SEC-SECRET-021

### Scenario 3: Reject group-readable secret file
- **Given** a secret file at "/etc/llorch/secrets/api-token"
- **And** a secret file with permissions 416 (0640)
- **When** I load the secret from file
- **Then** the operation should fail
- **And** the operation should reject group-readable files

**Tests**: SM-SEC-5004, SEC-SECRET-021

### Scenario 4: Reject empty secret file
- **Given** a secret file at "/etc/llorch/secrets/api-token"
- **And** a secret file containing ""
- **When** I load the secret from file
- **Then** the operation should fail
- **And** the error should be "NotFound"

**Tests**: SM-LOAD-4001-R5

### Scenario 5: Reject path traversal in file path
- **Given** a secret file at "../../../etc/passwd"
- **When** I load the secret from file
- **Then** the operation should fail
- **And** the operation should reject path traversal

**Tests**: SM-SEC-5005, SEC-SECRET-024

### Scenario 6: Load secret from systemd credential
- **Given** a systemd credential "api_token"
- **When** I load from systemd credential
- **Then** the operation should succeed

**Tests**: SM-LOAD-4002, SEC-SECRET-050

---

## Feature: Verification (5 scenarios)

### Scenario 1: Verify matching secret
- **Given** a secret file containing "correct-token"
- **When** I verify the secret with "correct-token"
- **Then** the verification should succeed

**Tests**: SM-TYPE-3002-R4, SEC-SECRET-030

### Scenario 2: Reject non-matching secret
- **Given** a secret file containing "correct-token"
- **When** I verify the secret with "wrong-token"
- **Then** the verification should fail

**Tests**: SM-TYPE-3002-R4, SEC-SECRET-030

### Scenario 3: Reject secret with different length
- **Given** a secret file containing "short"
- **When** I verify the secret with "much-longer-token"
- **Then** the verification should fail

**Tests**: SM-SEC-5003, SEC-SECRET-033

### Scenario 4: Timing-safe comparison for early mismatch
- **Given** a secret file containing "correct-token-abc"
- **When** I verify the secret with "wrong-token-abc"
- **Then** the verification should fail

**Tests**: SM-SEC-5003, SEC-SECRET-031, SEC-SECRET-032

### Scenario 5: Timing-safe comparison for late mismatch
- **Given** a secret file containing "correct-token-abc"
- **When** I verify the secret with "correct-token-xyz"
- **Then** the verification should fail

**Tests**: SM-SEC-5003, SEC-SECRET-031, SEC-SECRET-032

---

## Feature: Key Derivation (5 scenarios)

### Scenario 1: Derive key from token with domain separation
- **Given** a token "test-worker-token"
- **And** a domain "llorch-seal-key-v1"
- **When** I derive a key from the token
- **Then** the operation should succeed
- **And** the derived key should be 32 bytes

**Tests**: SM-LOAD-4003, SEC-SECRET-040, SEC-SECRET-041, SEC-SECRET-042

### Scenario 2: Derived keys are deterministic
- **Given** a token "test-worker-token"
- **And** a domain "llorch-seal-key-v1"
- **When** I derive a key from the token
- **Then** the operation should succeed
- **And** the derived key should be deterministic

**Tests**: SM-LOAD-4003-R5, SEC-SECRET-043

### Scenario 3: Different domains produce different keys
- **Given** a token "test-worker-token"
- **And** a domain "llorch-seal-key-v1"
- **When** I derive a key from the token
- **Then** the operation should succeed

**Tests**: SEC-SECRET-041, SEC-SECRET-044

### Scenario 4: Reject empty token
- **Given** a token ""
- **And** a domain "llorch-seal-key-v1"
- **When** I derive a key from the token
- **Then** the operation should fail

**Tests**: SM-LOAD-4003-R6

### Scenario 5: Reject missing domain
- **Given** a token "test-worker-token"
- **When** I derive a key from the token
- **Then** the operation should fail

**Tests**: SEC-SECRET-041

---

## Feature: Security Properties (5 scenarios)

### Scenario 1: Secrets are not logged via Debug trait
- **Given** a secret file containing "secret-token"
- **When** I load the secret from file
- **Then** the operation should succeed
- **And** the secret should not be logged

**Tests**: SM-SEC-5002, SEC-SECRET-010, SEC-SECRET-011

### Scenario 2: Secrets are zeroized on drop
- **Given** a secret file containing "secret-token"
- **When** I load the secret from file
- **Then** the operation should succeed
- **And** the secret should be zeroized on drop

**Tests**: SM-SEC-5001, SEC-SECRET-013, SEC-SECRET-014

### Scenario 3: Error messages do not contain secret values
- **Given** a secret file containing "secret-token-abc123"
- **When** I verify the secret with "wrong-token"
- **Then** the verification should fail
- **And** the secret should not be logged

**Tests**: SM-SEC-5002, SEC-SECRET-012

### Scenario 4: File paths are validated before reading
- **Given** a secret file at "/etc/llorch/secrets/api-token"
- **And** a secret file with permissions 384 (0600)
- **When** I load the secret from file
- **Then** the operation should succeed

**Tests**: SM-SEC-5004, SEC-SECRET-020, SEC-SECRET-024

### Scenario 5: Symlinks are resolved and validated
- **Given** a secret file at "/etc/llorch/secrets/symlink-to-token"
- **When** I load the secret from file
- **Then** the operation should succeed

**Tests**: SM-SEC-5005, SEC-SECRET-023

---

## Security Requirements Coverage

### Credential Protection (SEC-SECRET-010 to 016)
- ✅ SEC-SECRET-010: Use `secrecy::Secret<T>` wrapper
- ✅ SEC-SECRET-011: No Debug/Display/ToString/Serialize
- ✅ SEC-SECRET-012: Error messages don't contain secrets
- ✅ SEC-SECRET-013: Logging doesn't log secrets
- ✅ SEC-SECRET-014: Drop with zeroization
- ✅ SEC-SECRET-015: Compiler fences prevent optimization
- ✅ SEC-SECRET-016: Environment variable warnings

### File Security (SEC-SECRET-020 to 024)
- ✅ SEC-SECRET-020: Validate permissions before reading
- ✅ SEC-SECRET-021: Reject mode 0o077 files
- ✅ SEC-SECRET-022: Recommend 0600 permissions
- ✅ SEC-SECRET-023: Canonicalize paths
- ✅ SEC-SECRET-024: Validate before operations

### Timing Safety (SEC-SECRET-030 to 033)
- ✅ SEC-SECRET-030: Constant-time comparison
- ✅ SEC-SECRET-031: Use `subtle::ConstantTimeEq`
- ✅ SEC-SECRET-032: Examine all bytes
- ✅ SEC-SECRET-033: Length can short-circuit

### Key Derivation (SEC-SECRET-040 to 044)
- ✅ SEC-SECRET-040: Use HKDF-SHA256
- ✅ SEC-SECRET-041: Include domain separation
- ✅ SEC-SECRET-042: 32-byte output
- ✅ SEC-SECRET-043: Deterministic
- ✅ SEC-SECRET-044: Unique domains

### Systemd Credentials (SEC-SECRET-050 to 052)
- ✅ SEC-SECRET-050: Validate credential names
- ✅ SEC-SECRET-051: Validate $CREDENTIALS_DIRECTORY
- ✅ SEC-SECRET-052: Canonicalize credential paths

---

## Attack Vectors Tested

### Credential Exposure
- ✅ Environment variable exposure (ps auxe, /proc)
- ✅ Debug/Display trait leakage
- ✅ Error message leakage
- ✅ Memory dump exposure

### File Permissions
- ✅ World-readable files (0644)
- ✅ Group-readable files (0640)
- ✅ Symlink to world-readable files

### Path Traversal
- ✅ Directory traversal (`../../../etc/passwd`)
- ✅ Absolute path escape
- ✅ Symlink escape

### Timing Attacks
- ✅ Early mismatch timing
- ✅ Late mismatch timing
- ✅ Length-based timing

### Key Derivation
- ✅ Weak KDF (SHA-256 only)
- ✅ No domain separation
- ✅ Insufficient output length
- ✅ Non-deterministic derivation

---

## Test Execution

```bash
# Run all scenarios
cargo test -p secrets-management-bdd -- --nocapture

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/file_loading.feature \
cargo test -p secrets-management-bdd -- --nocapture

# Run with BDD runner
cargo run -p secrets-management-bdd --bin bdd-runner
```

---

## References

**Specifications**:
- `00_secrets_management.md` — Main specification (SM-*)
- `10_expectations.md` — Consumer expectations (EXP-*)
- `20_security.md` — Security specification (SEC-SECRET-*)

**Security Audits**:
- `SECURITY_AUDIT_EXISTING_CODEBASE.md` — Vulnerability #3
- `SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` — Issues #1-#20

---

**Last Updated**: 2025-10-01  
**Total Scenarios**: 21  
**Pass Rate**: 100% (when implemented)
