# TEAM-097: BDD P0 Security Tests

**Phase:** 1 - BDD Test Development  
**Duration:** 7-10 days  
**Priority:** P0 - Critical  
**Status:** 🔴 NOT STARTED

---

## Mission

Write comprehensive BDD tests for P0 security features:
1. Authentication (API token validation)
2. Secrets Management (file-based credentials)
3. Input Validation (injection prevention)

**Deliverable:** 45 BDD scenarios covering all P0 security requirements

---

## Assignments

### 1. Authentication Tests (15-20 scenarios)
**File:** `test-harness/bdd/tests/features/300-authentication.feature`

**Scenarios to Write:**
- [ ] AUTH-001: Reject request without Authorization header (401)
- [ ] AUTH-002: Reject request with invalid token (401)
- [ ] AUTH-003: Accept request with valid Bearer token (200)
- [ ] AUTH-004: Timing-safe token comparison (variance < 10%)
- [ ] AUTH-005: Loopback bind without token works (dev mode)
- [ ] AUTH-006: Public bind requires token or fails to start
- [ ] AUTH-007: Token fingerprinting in logs (never raw tokens)
- [ ] AUTH-008: Multiple components (queen, hive, worker) all require auth
- [ ] AUTH-009: Token validation on all HTTP endpoints
- [ ] AUTH-010: Invalid token format rejected
- [ ] AUTH-011: Empty token rejected
- [ ] AUTH-012: Token with special characters handled correctly
- [ ] AUTH-013: Concurrent auth requests (no race conditions)
- [ ] AUTH-014: Auth failure logged with fingerprint
- [ ] AUTH-015: Auth success logged with fingerprint

**Step Definitions Required:**
```rust
// test-harness/bdd/src/steps/authentication.rs
#[given("queen-rbee is running with auth enabled")]
#[when("I send POST /v1/workers/spawn without Authorization header")]
#[then("response status is 401 Unauthorized")]
#[then("response body contains {string}")]
#[then("log contains token fingerprint (not raw token)")]
#[then("response times have variance < {int}%")]
```

---

### 2. Secrets Management Tests (10-15 scenarios)
**File:** `test-harness/bdd/tests/features/310-secrets-management.feature`

**Scenarios to Write:**
- [ ] SEC-001: Load API token from file with 0600 permissions
- [ ] SEC-002: Reject world-readable secret file (0644)
- [ ] SEC-003: Reject group-readable secret file (0640)
- [ ] SEC-004: Load from systemd credentials (/run/credentials/)
- [ ] SEC-005: Memory zeroization on drop (not in memory dump)
- [ ] SEC-006: Derive encryption key from token (HKDF-SHA256)
- [ ] SEC-007: Secret file must exist or fail to start
- [ ] SEC-008: Secret file must be readable or fail to start
- [ ] SEC-009: Secrets never appear in logs
- [ ] SEC-010: Secrets never appear in error messages
- [ ] SEC-011: Timing-safe secret verification
- [ ] SEC-012: Secret rotation without restart (SIGHUP)

**Step Definitions Required:**
```rust
// test-harness/bdd/src/steps/secrets.rs
#[given("API token file exists at {string}")]
#[given("file permissions are {string}")]
#[when("queen-rbee starts")]
#[then("API token is loaded from file")]
#[then("token is stored in memory with zeroization")]
#[then("token is never logged")]
#[then("queen-rbee fails to start")]
#[then("displays error: {string}")]
```

---

### 3. Input Validation Tests (20-25 scenarios)
**File:** Expand `test-harness/bdd/tests/features/140-input-validation.feature`

**Scenarios to Add:**
- [ ] VAL-001: Prevent log injection with newlines
- [ ] VAL-002: Prevent log injection with ANSI escape codes
- [ ] VAL-003: Prevent log injection with null bytes
- [ ] VAL-004: Prevent path traversal (../../etc/passwd)
- [ ] VAL-005: Prevent path traversal (absolute paths)
- [ ] VAL-006: Prevent path traversal (symlinks)
- [ ] VAL-007: Prevent command injection (shell metacharacters)
- [ ] VAL-008: Prevent command injection (backticks)
- [ ] VAL-009: Prevent command injection (pipe characters)
- [ ] VAL-010: Validate model reference format (hf:org/repo)
- [ ] VAL-011: Validate model reference (no special chars)
- [ ] VAL-012: Validate worker ID format (alphanumeric + dash)
- [ ] VAL-013: Validate worker ID length (max 64 chars)
- [ ] VAL-014: Validate backend name (cuda, metal, cpu only)
- [ ] VAL-015: Validate device ID (non-negative integer)
- [ ] VAL-016: Validate port number (1024-65535)
- [ ] VAL-017: Validate node name (DNS-safe characters)
- [ ] VAL-018: Reject SQL injection attempts (if applicable)
- [ ] VAL-019: Reject XSS attempts (if applicable)
- [ ] VAL-020: Property-based testing (proptest fuzzing)

**Step Definitions Required:**
```rust
// test-harness/bdd/src/steps/validation.rs
#[when("I send request with log injection payload {string}")]
#[when("I send request with path traversal payload {string}")]
#[when("I send request with command injection payload {string}")]
#[then("request is rejected with 400 Bad Request")]
#[then("error message is safe (no payload echoed)")]
#[then("validation error explains expected format")]
```

---

## Deliverables

### Feature Files
- [ ] `300-authentication.feature` (15-20 scenarios)
- [ ] `310-secrets-management.feature` (10-15 scenarios)
- [ ] `140-input-validation.feature` (expand with 20-25 scenarios)

### Step Definitions
- [ ] `src/steps/authentication.rs` (all auth steps)
- [ ] `src/steps/secrets.rs` (all secrets steps)
- [ ] `src/steps/validation.rs` (all validation steps)

### Documentation
- [ ] Update `test-harness/bdd/README.md` with new features
- [ ] Document test coverage in this file
- [ ] Create handoff document (≤ 2 pages)

---

## Acceptance Criteria

### For Each Feature File
- [ ] All scenarios follow Given-When-Then structure
- [ ] Scenarios are independent (no shared state)
- [ ] Tags are appropriate (@p0, @auth, @secrets, @validation)
- [ ] Traceability to RC checklist item
- [ ] Background section defines topology
- [ ] Examples use realistic data

### For Step Definitions
- [ ] Import REAL product code from `/bin/`
- [ ] No mocks for core functionality
- [ ] Error handling for all steps
- [ ] Clear assertion messages
- [ ] Reusable across scenarios

### For Test Suite
- [ ] All 45 scenarios run without errors
- [ ] Tests fail against current code (expected - no implementation yet)
- [ ] No compilation errors
- [ ] No unwrap() in step definitions
- [ ] Coverage report generated

---

## Testing Commands

```bash
# Run all security tests
LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/300-authentication.feature \
  cargo run --bin bdd-runner

LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/310-secrets-management.feature \
  cargo run --bin bdd-runner

LLORCH_BDD_FEATURE_PATH=test-harness/bdd/tests/features/140-input-validation.feature \
  cargo run --bin bdd-runner

# Run with tags
cargo run --bin bdd-runner -- --tags @p0
cargo run --bin bdd-runner -- --tags @auth
cargo run --bin bdd-runner -- --tags @secrets
cargo run --bin bdd-runner -- --tags @validation

# Generate coverage report
cargo tarpaulin --bin bdd-runner --out Html
```

---

## Progress Tracking

### Day 1-2: Authentication Tests
- [ ] Write 300-authentication.feature (15-20 scenarios)
- [ ] Implement authentication step definitions
- [ ] Run tests (expect failures)
- [ ] Document coverage

### Day 3-4: Secrets Management Tests
- [ ] Write 310-secrets-management.feature (10-15 scenarios)
- [ ] Implement secrets step definitions
- [ ] Run tests (expect failures)
- [ ] Document coverage

### Day 5-7: Input Validation Tests
- [ ] Expand 140-input-validation.feature (20-25 scenarios)
- [ ] Implement validation step definitions
- [ ] Run tests (expect failures)
- [ ] Document coverage

### Day 8-10: Integration & Handoff
- [ ] Run all security tests together
- [ ] Generate coverage report
- [ ] Update documentation
- [ ] Create handoff document
- [ ] Verify 45 scenarios complete

---

## Checklist

**Feature Files:**
- [x] 300-authentication.feature (15-20 scenarios) ✅ COMPLETE (20 scenarios)
- [x] 310-secrets-management.feature (10-15 scenarios) ✅ COMPLETE (17 scenarios)
- [x] 140-input-validation.feature expanded (20-25 scenarios) ✅ COMPLETE (25 scenarios)

**Step Definitions:**
- [x] authentication.rs ✅ COMPLETE (40+ functions)
- [x] secrets.rs ✅ COMPLETE (35+ functions)
- [x] validation.rs ✅ COMPLETE (30+ functions)

**Documentation:**
- [x] README.md updated ✅ COMPLETE
- [x] Coverage documented ✅ COMPLETE
- [x] Handoff created ✅ COMPLETE

**Completion:** 62/62 scenarios (100%) ✅

**Feature Files Created:**
- ✅ 300-authentication.feature (20 scenarios)
- ✅ 310-secrets-management.feature (17 scenarios)
- ✅ 140-input-validation.feature (25 new scenarios added)

**Step Definitions Created:**
- ✅ authentication.rs (40+ step functions)
- ✅ secrets.rs (35+ step functions)
- ✅ validation.rs (30+ step functions)

**Compilation:** ✅ PASS (`cargo check --bin bdd-runner`)

---

## Notes for Next Team (TEAM-098)

- All security tests should be failing (no implementation yet)
- Step definitions use real code from `/bin/`
- No mocks for auth-min, secrets-management, input-validation crates
- Tests are ready for implementation teams to make pass

---

## References

- `RELEASE_CANDIDATE_CHECKLIST.md` - RC requirements
- `BDD_TESTS_FOR_RC_CHECKLIST.md` - Test specifications
- `SHARED_CRATES.md` - Available security crates
- `test-harness/bdd/README.md` - BDD testing guide
- `.windsurf/rules/engineering-rules.md` - Mandatory rules

---

**Created by:** TEAM-096 | 2025-10-18  
**Assigned to:** TEAM-097  
**Status:** Ready to start  
**Next Team:** TEAM-098 (BDD P0 Lifecycle Tests)
