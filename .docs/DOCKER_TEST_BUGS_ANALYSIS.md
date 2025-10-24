# Docker Test Bugs & Wrong Assumptions Analysis

**Date:** Oct 24, 2025  
**Status:** 🔍 ANALYSIS COMPLETE

---

## Bugs Found

### 1. **SSH Tests Use docker exec, Not Real SSH** ❌

**Location:** All `ssh_communication_tests.rs`

**Problem:**
```rust
// Test comment says "SSH" but actually uses docker exec
let output = harness.exec("rbee-hive-localhost", &["echo", "test"]).await
```

**Wrong Assumption:** These tests claim to test SSH but actually test `docker exec`, which is NOT the same as SSH.

**Impact:**
- Tests don't validate real SSH connections
- Tests don't validate SSH authentication
- Tests don't validate SSH key exchange
- Tests don't validate network-level SSH communication

**Real SSH would be:**
```rust
let mut client = RbeeSSHClient::connect("localhost", 2222, "rbee").await?;
let (stdout, _, exit_code) = client.exec("echo test").await?;
```

---

### 2. **Concurrent Test Has Lifetime Issue** ❌

**Location:** `ssh_communication_tests.rs:116`

**Problem:**
```rust
for i in 0..5 {
    let harness_clone = &harness;  // ❌ Borrow doesn't live long enough
    let handle = tokio::spawn(async move {
        harness_clone.exec(...).await  // ❌ Can't move borrowed reference
    });
}
```

**Error:** This won't compile! Can't move a borrowed reference into `tokio::spawn`.

**Fix Needed:** Use `Arc<DockerTestHarness>` or restructure.

---

### 3. **Deprecated format! in expect()** ⚠️

**Location:** Multiple files

**Problem:**
```rust
.expect(&format!("Request {} failed", i))  // ❌ Allocates String unnecessarily
```

**Better:**
```rust
.unwrap_or_else(|e| panic!("Request {} failed: {}", i, e))
```

---

### 4. **Tests Assume Binaries Exist in Container** ❌

**Location:** `docker_smoke_test.rs:92`, `ssh_communication_tests.rs:55`

**Problem:**
```rust
.exec("rbee-hive-localhost", &["ls", "-la", "/home/rbee/.local/bin/rbee-hive"])
```

**Wrong Assumption:** The Dockerfile copies pre-built binaries, but tests assume they exist.

**Reality Check:** Looking at `Dockerfile.hive`:
```dockerfile
COPY --chown=rbee:rbee target/debug/rbee-hive /home/rbee/.local/bin/rbee-hive
```

This requires `target/debug/rbee-hive` to exist BEFORE building the image!

**Impact:** Tests will fail if binaries aren't built first.

---

### 5. **Health Check Response Assumption** ⚠️

**Location:** Multiple files

**Problem:**
```rust
assert_eq!(response.into_string().unwrap(), "ok");
```

**Wrong Assumption:** Health endpoint returns exactly "ok".

**Reality Check:** Need to verify actual health endpoint implementation.

---

### 6. **Capabilities Response Structure Assumption** ⚠️

**Location:** `http_communication_tests.rs:45-48`

**Problem:**
```rust
assert!(first_device["id"].is_string());
assert!(first_device["name"].is_string());
assert!(first_device["device_type"].is_string());
```

**Wrong Assumption:** Assumes specific JSON structure without checking actual implementation.

---

### 7. **Connection Refused Test Flakiness** ⚠️

**Location:** `http_communication_tests.rs:76-87`

**Problem:**
```rust
async fn test_http_connection_refused() {
    // Don't start harness - no containers running
    let result = ureq::get("http://localhost:9000/health")
        .timeout(Duration::from_secs(2))
        .call();
    assert!(result.is_err(), "Connection should be refused when no hive running");
}
```

**Wrong Assumption:** Assumes port 9000 is free.

**Reality:** If another test is running or port is in use, this test fails.

---

### 8. **Missing Error Context** ⚠️

**Location:** All test files

**Problem:**
```rust
.unwrap()  // ❌ No context on failure
```

**Better:**
```rust
.expect("Failed to connect to hive")  // ✅ Clear error message
```

---

### 9. **Test Isolation Issues** ❌

**Location:** All tests

**Problem:** Multiple tests use same ports (8500, 9000) without isolation.

**Impact:** Tests can't run in parallel, port conflicts possible.

---

### 10. **Missing Cleanup Verification** ⚠️

**Location:** `DockerTestHarness::drop()`

**Problem:** Drop doesn't verify cleanup succeeded.

**Impact:** Failed cleanup leaves containers running, polluting environment.

---

## Summary

| Bug | Severity | Impact | Fix Complexity |
|-----|----------|--------|----------------|
| SSH tests use docker exec | High | Tests don't validate SSH | Medium |
| Concurrent test lifetime issue | High | Won't compile | Low |
| Binary existence assumption | High | Tests fail if not built | Low (docs) |
| format! in expect() | Low | Performance | Low |
| Health response assumption | Medium | May fail | Low (verify) |
| Capabilities structure assumption | Medium | May fail | Low (verify) |
| Connection refused flakiness | Medium | Flaky tests | Medium |
| Missing error context | Low | Hard to debug | Low |
| Test isolation | Medium | Can't run parallel | High |
| Missing cleanup verification | Low | Polluted environment | Low |

---

## Recommendations

### Priority 1: Fix Compilation Issues
1. Fix concurrent test lifetime issue
2. Verify tests actually compile

### Priority 2: Fix Wrong Assumptions
3. Document that binaries must be built first
4. Verify health endpoint response format
5. Verify capabilities JSON structure
6. Either use real SSH or rename tests to "docker exec tests"

### Priority 3: Improve Test Quality
7. Add error context to all unwrap() calls
8. Fix connection refused test to ensure port is free
9. Add cleanup verification
10. Consider test isolation strategy

---

## Next Steps

1. **Verify actual implementations** - Check what health/capabilities actually return
2. **Fix compilation errors** - Fix lifetime issues
3. **Update documentation** - Clarify what tests actually test
4. **Add real SSH tests** - If SSH testing is needed
5. **Improve error messages** - Add context to failures
