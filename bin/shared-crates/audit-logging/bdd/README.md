# audit-logging-bdd

**BDD test suite for audit-logging with security-focused validation**

`bin/shared-crates/audit-logging/bdd` — Cucumber-based behavior-driven development tests for the audit-logging security crate.

---

## What This Crate Does

This is the **BDD test harness** for audit-logging. It provides:

- **Cucumber/Gherkin scenarios** testing audit event validation
- **Security-focused BDD tests** for log injection prevention
- **Comprehensive attack surface coverage** for all event types
- **Maximum robustness testing** with edge cases and boundary conditions

**Test Coverage**:
- ✅ Authentication events (AuthSuccess, AuthFailure, TokenCreated, TokenRevoked)
- ✅ Resource operation events (PoolCreated, PoolDeleted, TaskSubmitted, etc.)
- ✅ VRAM operation events (VramSealed, SealVerified, VramAllocated, etc.)
- ✅ Security incident events (PathTraversalAttempt, PolicyViolation, etc.)
- ✅ Event serialization and data integrity

---

## Running Tests

### All Scenarios

```bash
# Run all BDD tests
cargo test -p audit-logging-bdd -- --nocapture

# Or use the BDD runner binary
cargo run -p audit-logging-bdd --bin bdd-runner
```

### Specific Feature

```bash
# Set environment variable to target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/authentication_events.feature \
cargo test -p audit-logging-bdd -- --nocapture
```

---

## Feature Files

Located in `tests/features/`:

- `authentication_events.feature` — Authentication event validation
- `resource_events.feature` — Resource operation event validation
- `vram_events.feature` — VRAM operation event validation
- `security_events.feature` — Security incident event validation
- `event_serialization.feature` — JSON serialization tests

---

## Example Scenario

```gherkin
Feature: Authentication Event Validation

  Scenario: Reject AuthSuccess with ANSI escape in user ID
    Given a user ID "\x1b[31mFAKE ERROR\x1b[0m"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Accept valid AuthSuccess event
    Given a user ID "admin@example.com"
    And a path "/v2/tasks"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should succeed
```

---

## Step Definitions

Located in `src/steps/`:

- `world.rs` — BDD world state (events, actors, validation results)
- `validation.rs` — Validation action steps (Given/When)
- `assertions.rs` — Assertion steps (Then)

---

## Testing

```bash
# Run all tests
cargo test -p audit-logging-bdd -- --nocapture

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/security_events.feature \
cargo test -p audit-logging-bdd -- --nocapture
```

---

## Dependencies

### Parent Crate

- `audit-logging` — The library being tested

### Test Infrastructure

- `cucumber` — BDD framework
- `tokio` — Async runtime for tests
- `anyhow` — Error handling
- `chrono` — Timestamp handling
- `serde_json` — Event serialization

---

## Specifications

Tests verify requirements from:
- **AUDIT-SEC-001 to AUDIT-SEC-023**: Security requirements
- **AUDIT-VAL-001 to AUDIT-VAL-015**: Input validation requirements
- **SECURITY_AUDIT_EXISTING_CODEBASE.md**: Vulnerability #18 (Log Injection)
- **.specs/20_security.md**: Complete attack surface analysis

See `.specs/README.md` and `.specs/20_security.md` for full requirements.

---

## Security Attack Vectors Tested

### Log Injection Prevention

All BDD scenarios test against real-world attack vectors:

- ✅ **ANSI escape injection** - Terminal manipulation (`\x1b[31m`, `\x1b[0m`)
- ✅ **Control character injection** - Log line injection (`\r`, `\n`, `\t`)
- ✅ **Null byte injection** - C string truncation (`\0`)
- ✅ **Unicode attacks** - Directional overrides (`\u{202E}`, `\u{202D}`)
- ✅ **Path traversal** - Directory traversal (`../`, `..\`)
- ✅ **Log line injection** - Fake log entries via newlines
- ✅ **Terminal manipulation** - ANSI escapes, control chars
- ✅ **Display spoofing** - Unicode directional overrides

### Escape Sequence Support

BDD scenarios support testing with special characters:

- `\n` - Newline (0x0A)
- `\r` - Carriage return (0x0D)
- `\t` - Tab (0x09)
- `\0` - Null byte (0x00)
- `\x1b` - ANSI escape (0x1B)
- `\x01` - Control character (0x01)
- `\x07` - Bell (0x07)
- `\x08` - Backspace (0x08)
- `\x0b` - Vertical tab (0x0B)
- `\x0c` - Form feed (0x0C)
- `\x1f` - Control character (0x1F)

---

## Event Types Tested

### Authentication Events (4 types)

- `AuthSuccess` — Successful authentication
- `AuthFailure` — Failed authentication attempt
- `TokenCreated` — API token created
- `TokenRevoked` — API token revoked

### Resource Operation Events (8 types)

- `PoolCreated` — Pool created
- `PoolDeleted` — Pool deleted
- `PoolModified` — Pool modified
- `NodeRegistered` — Node registered
- `NodeDeregistered` — Node deregistered
- `TaskSubmitted` — Task submitted
- `TaskCompleted` — Task completed
- `TaskCanceled` — Task canceled

### VRAM Operation Events (6 types)

- `VramSealed` — Model sealed in VRAM
- `SealVerified` — Seal verified successfully
- `SealVerificationFailed` — Seal verification failed
- `VramAllocated` — VRAM allocated
- `VramAllocationFailed` — VRAM allocation failed
- `VramDeallocated` — VRAM deallocated

### Security Incident Events (5 types)

- `RateLimitExceeded` — Rate limit exceeded
- `PathTraversalAttempt` — Path traversal attempt
- `InvalidTokenUsed` — Invalid token used
- `PolicyViolation` — Security policy violation
- `SuspiciousActivity` — Suspicious activity detected

**Total**: 32 event types across 7 categories

---

## Robustness Features

### Input Validation Coverage

- ✅ **All string fields validated** - user_id, resource_id, reason, details, etc.
- ✅ **Length limits enforced** - Max 1024 chars per field
- ✅ **Control character removal** - Except newline in structured fields
- ✅ **ANSI escape removal** - All terminal control sequences
- ✅ **Null byte removal** - Prevents C string truncation
- ✅ **Unicode normalization** - Removes directional overrides

### Test Quality Metrics

- ✅ **Security-first testing** - All attack vectors covered
- ✅ **Fast execution** - Async tests with tokio
- ✅ **Comprehensive coverage** - All 32 event types
- ✅ **Real-world scenarios** - Based on actual attack patterns

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha - **Security-critical validation** with maximum robustness
- **Security Tier**: TIER 1 (security-critical) ✅ **MAINTAINED**
- **Maintainers**: @llama-orch-maintainers

---

## Adding New Tests

1. Create or edit `.feature` files under `tests/features/`
2. Implement step definitions in `src/steps/` if needed
3. Run tests to verify

Example:

```gherkin
# tests/features/my_new_test.feature
Feature: New Event Validation Test

  Scenario: Test something
    Given a user ID "test@example.com"
    And a path "/v2/test"
    When I create an AuthSuccess event
    And I validate the event
    Then the validation should succeed
```

---

## Integration with Parent Crate

This BDD suite tests the **validation module** of the audit-logging crate:

```rust
// In audit-logging/src/validation.rs
pub fn validate_event(event: &mut AuditEvent) -> Result<()> {
    // Validates and sanitizes all user-controlled fields
    // Prevents log injection, ANSI escapes, control chars, etc.
}
```

The BDD tests ensure that:
1. Valid events pass validation
2. Malicious input is rejected
3. Sanitization works correctly
4. All 32 event types are covered

---

**For questions**: See `.specs/README.md` and `.docs/testing/BDD_WIRING.md`
