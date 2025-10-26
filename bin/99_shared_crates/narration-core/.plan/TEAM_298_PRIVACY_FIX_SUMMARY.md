# TEAM-298: Privacy Fix Summary

**Date:** 2025-10-26  
**Severity:** CRITICAL  
**Status:** MUST IMPLEMENT IMMEDIATELY

---

## What We Discovered

TEAM-297 and TEAM-298 implemented Phases 0 & 1, but there's a **CRITICAL privacy violation**:

```rust
// Current implementation (BROKEN):
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
```

**This prints ALL narration to global stderr!**

### The Problem

In a multi-tenant environment:
- User A submits job with secret data
- User B submits different job
- **Both see each other's narration on stderr!**

This is a **security violation** and **privacy leak**.

---

## What Needs to Change

### 1. Remove Global stderr (TEAM-298)

**Current (BROKEN):**
```rust
// Always prints to stderr
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
```

**Fixed (SECURE):**
```rust
// Only print in keeper mode (single-user)
if is_keeper_mode() {
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
}
```

### 2. Add Keeper Mode Flag (TEAM-298)

```rust
// In src/lib.rs
static KEEPER_MODE: Lazy<bool> = Lazy::new(|| {
    std::env::var("RBEE_KEEPER_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
});

pub fn is_keeper_mode() -> bool {
    *KEEPER_MODE
}
```

### 3. SSE Becomes Primary (TEAM-298)

**Philosophy change:**
- **Before:** stderr primary, SSE bonus
- **After:** SSE primary (job-scoped), stderr only in keeper mode

### 4. Keeper Sets Mode (TEAM-301)

```rust
// In keeper main()
fn main() -> Result<()> {
    // Enable keeper mode (single-user CLI)
    std::env::set_var("RBEE_KEEPER_MODE", "1");
    
    // Now n!() will print to terminal
    // ...
}
```

---

## Security Model

### Multi-User Daemons (queen, hive, worker)

```
RBEE_KEEPER_MODE=0  // Default (secure)

Narration flow:
  n!("action", "msg")
    → Has job_id? → SSE channel (isolated) ✅
    → No job_id? → DROPPED (security) ✅
    → stderr? → NEVER (privacy) ✅
```

### Single-User CLI (keeper)

```
RBEE_KEEPER_MODE=1  // Set by keeper

Narration flow:
  n!("action", "msg")
    → Has job_id? → SSE channel (if available)
    → stderr? → YES (user's own terminal) ✅
```

---

## Implementation Checklist

### TEAM-298 (IMMEDIATE):
- [ ] Add `RBEE_KEEPER_MODE` environment variable
- [ ] Add `is_keeper_mode()` function
- [ ] Make stderr conditional on keeper mode
- [ ] Update `narrate_at_level()` to check keeper mode
- [ ] Add multi-tenant isolation tests
- [ ] Add keeper mode tests
- [ ] Update existing tests (enable keeper mode or use capture)
- [ ] Document privacy fix in handoff

### TEAM-299:
- [ ] Verify daemons never set keeper mode
- [ ] Test SSE-only operation in daemons
- [ ] Document security model

### TEAM-301:
- [ ] Set `RBEE_KEEPER_MODE=1` in keeper main()
- [ ] Test terminal output in keeper
- [ ] Verify single-user experience

---

## Testing Strategy

### Privacy Tests (CRITICAL!)

```rust
#[tokio::test]
async fn test_multi_tenant_isolation() {
    // Create two separate job channels
    let job_a = "user-a-job";
    let job_b = "user-b-job";
    
    // Verify User A never sees User B's data
    // Verify User B never sees User A's data
}

#[test]
fn test_no_stderr_in_multi_user_mode() {
    // Ensure keeper mode OFF
    std::env::remove_var("RBEE_KEEPER_MODE");
    
    // Verify no stderr output
    n!("test", "Should not print");
}

#[test]
fn test_stderr_in_keeper_mode() {
    // Enable keeper mode
    std::env::set_var("RBEE_KEEPER_MODE", "1");
    
    // Verify stderr output works
    n!("test", "Can print");
}
```

---

## Migration Impact

### Existing Tests

**Need to update:**
- Tests that expect stderr output
- Tests that don't set keeper mode

**Solutions:**
1. Enable keeper mode in tests: `std::env::set_var("RBEE_KEEPER_MODE", "1");`
2. Use capture adapter: `CaptureAdapter::install()`

### Existing Code

**No changes required!**

The `n!()` macro works the same way:
```rust
n!("action", "message");
```

Only difference:
- **Before:** Always printed to stderr
- **After:** Only printed in keeper mode (secure!)

---

## Documentation Updates

### Files Updated:
- [x] PRIVACY_FIX_REQUIRED.md - Comprehensive privacy fix document
- [x] MASTERPLAN.md - Added privacy fix to Phase 1
- [x] README.md - Added privacy fix to Phase 1 summary
- [x] TEAM_298_PHASE_1_SSE_OPTIONAL.md - Added privacy fix tasks
- [x] TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md - Added keeper mode setup

### Files to Create (TEAM-298):
- [ ] tests/privacy_isolation_tests.rs - Multi-tenant isolation tests
- [ ] TEAM_298_HANDOFF.md - Document privacy fix implementation

---

## Risk Assessment

### If Not Fixed:
- ❌ **HIGH RISK:** Privacy violations in production
- ❌ **HIGH RISK:** Sensitive data exposure
- ❌ **HIGH RISK:** Compliance issues (GDPR, etc.)
- ❌ **HIGH RISK:** Security audit failures

### With Fix:
- ✅ **LOW RISK:** Job-scoped narration (SSE)
- ✅ **LOW RISK:** No cross-job leaks
- ✅ **LOW RISK:** Keeper mode for single-user
- ✅ **LOW RISK:** Secure by default

---

## Timeline

### Week 2 (TEAM-298) - IMMEDIATE:
- Days 1-2: Implement keeper mode flag
- Days 3-4: Update narrate_at_level() with privacy fix
- Day 5: Add privacy tests, verify isolation

### Week 5 (TEAM-301):
- Day 1: Set keeper mode in keeper main()
- Days 2-3: Test terminal output
- Days 4-5: Integration testing

---

## Success Criteria

✅ **Privacy:**
- Multi-user daemons NEVER print to stderr
- Job narration isolated to SSE channels
- No cross-job data leaks

✅ **Security:**
- job_id required for SSE routing
- No job_id = dropped (not printed)
- Fail-fast security model

✅ **Usability:**
- Keeper can display output (single-user)
- Users see their own jobs only
- No privacy violations

✅ **Testing:**
- Multi-tenant isolation verified
- Keeper mode tested
- No regressions

---

## Conclusion

**The privacy violation is CRITICAL and must be fixed immediately.**

**TEAM-298 must implement the keeper mode flag and conditional stderr.**

**This is NOT optional - it's a security requirement!**

See [PRIVACY_FIX_REQUIRED.md](./PRIVACY_FIX_REQUIRED.md) for full details.
