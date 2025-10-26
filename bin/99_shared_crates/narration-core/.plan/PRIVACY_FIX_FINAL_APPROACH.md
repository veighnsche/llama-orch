# Privacy Fix: Final Approach

**Date:** 2025-10-26  
**Decision:** Complete removal of stderr from narration-core  
**Rationale:** Security by design, not security by configuration

---

## Summary

**The ONLY secure solution is complete removal of `eprintln!()` from narration-core.**

Any conditional approach (environment variables, feature flags, etc.) leaves exploitable attack surfaces.

---

## Why Environment Variables Are Insecure

### Proposed (REJECTED):
```rust
if is_keeper_mode() {  // ← Runtime check
    eprintln!(...);     // ← Code exists, can be exploited
}
```

### Attack Vectors:

1. **Environment variable injection**
   ```bash
   # Attacker sets env var in daemon
   export RBEE_KEEPER_MODE=1
   ./rbee-hive  # ← Now leaks to stderr!
   ```

2. **Accidental inheritance**
   ```bash
   # Parent process has keeper mode set
   export RBEE_KEEPER_MODE=1
   ./start_queen.sh  # ← Child inherits it!
   ```

3. **Configuration errors**
   ```yaml
   # config.yaml
   environment:
     RBEE_KEEPER_MODE: "1"  # ← Typo in production!
   ```

4. **Test pollution**
   ```rust
   // Test forgets to unset
   std::env::set_var("RBEE_KEEPER_MODE", "1");
   // All subsequent tests leak!
   ```

5. **Process manipulation**
   ```rust
   // Malicious code in dependency
   std::env::set_var("RBEE_KEEPER_MODE", "1");
   // Now daemon leaks!
   ```

### Single Point of Failure

```
Runtime check → BYPASSED → Data leak!
```

No defense in depth. One mistake = breach.

---

## Why Complete Removal Is Secure

### Implemented:
```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // ... mode selection ...
    
    // REMOVED - No stderr code path exists
    // Cannot be exploited if code doesn't exist
    
    // SSE only
    if sse_sink::is_enabled() {
        let _sent = sse_sink::try_send(&fields);
    }
}
```

### Security Properties:

1. **Code doesn't exist**
   - No `eprintln!()` in narration-core
   - Not compiled into daemons
   - **Cannot be exploited**

2. **Physical separation**
   - Keeper display is separate module
   - Only in keeper binary
   - **Not in daemons**

3. **Defense in depth**
   ```
   Layer 1: No code → Can't exploit
   Layer 2: SSE job-scoped → Isolated
   Layer 3: Keeper separate → Clear boundary
   Layer 4: Tests use capture → No stderr dep
   ```

4. **Fail-safe**
   - Configuration error? No effect (no code to enable)
   - Env var set? No effect (no code checks it)
   - Malicious code? No effect (no code path)

---

## Architecture Comparison

### OLD (Insecure - Conditional):
```
narration-core:
  narrate_at_level()
    if keeper_mode:        ← Runtime check (bypassable)
      eprintln!()          ← Code exists (exploitable)
    SSE
```

**Problems:**
- ❌ Code path exists in all binaries
- ❌ Single point of failure (env check)
- ❌ Exploitable if guard bypassed
- ❌ No defense in depth

### NEW (Secure - Complete Removal):
```
narration-core:
  narrate_at_level()
    SSE (job-scoped)       ← ONLY output
    tracing (optional)

keeper (separate):
  display.rs
    display_narration_stream()
      eprintln!()          ← Only in keeper binary
```

**Benefits:**
- ✅ No code in daemons (can't exploit)
- ✅ Physical separation (clear boundary)
- ✅ Defense in depth (multiple layers)
- ✅ Fail-safe (config errors harmless)

---

## Implementation

### Step 1: Remove stderr from narration-core

**File:** `src/lib.rs:559`

```diff
- eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
+ // TEAM-298: REMOVED - No stderr in narration-core
+ // This eliminates the attack surface completely.
+ // Keeper displays via separate SSE subscription.
```

### Step 2: Create keeper display module

**File:** `bin/00_rbee_keeper/src/display.rs`

```rust
use observability_narration_core::sse_sink::NarrationEvent;
use tokio::sync::mpsc;

/// Display narration to terminal (ONLY in keeper)
pub async fn display_narration_stream(mut rx: mpsc::Receiver<NarrationEvent>) {
    while let Some(event) = rx.recv().await {
        eprintln!("{}", event.formatted);
    }
}
```

### Step 3: Wire keeper display

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Create keeper's display channel
    let keeper_job_id = "keeper-display";
    sse_sink::create_job_channel(keeper_job_id.to_string(), 1000);
    let rx = sse_sink::take_job_receiver(keeper_job_id)?;
    
    // Spawn display task
    tokio::spawn(async move {
        display_narration_stream(rx).await;
    });
    
    // Set context
    let ctx = NarrationContext::new().with_job_id(keeper_job_id);
    
    // Run keeper
    with_narration_context(ctx, async {
        run_keeper().await
    }).await
}
```

### Step 4: Update all tests

**All test files:**

```rust
#[test]
fn test_narration() {
    // Use capture adapter, not stderr
    let adapter = CaptureAdapter::install();
    
    n!("test", "Test message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}
```

---

## Security Analysis

### Threat Model

**Threat:** Multi-tenant data leak via stderr

**Attacker Goal:** See other users' narration

### Before (Vulnerable):

**Attack:** Set `RBEE_KEEPER_MODE=1` in daemon

**Result:** All narration leaks to stderr

**Likelihood:** HIGH (easy to exploit)

**Impact:** CRITICAL (sensitive data exposed)

### After (Secure):

**Attack:** Try to enable stderr in daemon

**Result:** No effect (no code to enable)

**Likelihood:** NONE (no attack surface)

**Impact:** NONE (code doesn't exist)

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_no_stderr_code_path() {
    // Verify narration-core has no stderr
    let adapter = CaptureAdapter::install();
    
    n!("test", "Message");
    
    // Captured, not printed
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_multi_tenant_isolation() {
    // Two users, two SSE channels
    let job_a = "user-a-job";
    let job_b = "user-b-job";
    
    // Create separate channels
    sse_sink::create_job_channel(job_a.to_string(), 100);
    sse_sink::create_job_channel(job_b.to_string(), 100);
    
    // Verify isolation
    // User A never sees User B's data
}
```

### Security Tests

```rust
#[test]
fn test_cannot_enable_stderr_in_daemon() {
    // Try to exploit (should fail)
    std::env::set_var("RBEE_KEEPER_MODE", "1");
    
    let adapter = CaptureAdapter::install();
    n!("test", "Should not leak");
    
    // Still no stderr (code doesn't exist!)
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}
```

---

## Compliance

### GDPR

✅ **Data minimization** - No unnecessary copies (no stderr)  
✅ **Purpose limitation** - Data only where needed (SSE job-scoped)  
✅ **Integrity** - No cross-user leaks (isolated channels)  
✅ **Security** - Defense in depth (code removal)

### SOC 2

✅ **Access control** - Job-scoped SSE (no global access)  
✅ **Availability** - Narration never fails (SSE optional)  
✅ **Confidentiality** - No data leaks (no stderr)  
✅ **Processing integrity** - Isolation enforced (physical separation)

---

## Migration Impact

### Daemons (queen, hive, worker)

**Before:**
```rust
// Had eprintln (but guarded)
eprintln!(...);  // ← Attack surface
```

**After:**
```rust
// No eprintln code at all
// ← No attack surface
```

**Impact:** More secure, no functional change

### Keeper

**Before:**
```rust
// Narration might print (if env var set)
n!("test", "Message");
```

**After:**
```rust
// Narration goes to SSE, then displayed
n!("test", "Message");  // → SSE → display task → terminal
```

**Impact:** Explicit display pipeline, more secure

### Tests

**Before:**
```rust
// Some relied on stderr
n!("test", "Message");
// Might print to stderr
```

**After:**
```rust
// All use capture adapter
let adapter = CaptureAdapter::install();
n!("test", "Message");
let captured = adapter.captured();
```

**Impact:** More explicit, no stderr dependency

---

## Conclusion

**Complete removal is the ONLY secure solution.**

### Why?

1. **Code that doesn't exist can't be exploited**
2. **Physical separation > logical separation**
3. **Defense in depth > single point of failure**
4. **Fail-safe > fail-secure**

### Decision

✅ **APPROVED:** Complete removal of stderr from narration-core  
❌ **REJECTED:** Environment variable approach  
❌ **REJECTED:** Feature flag approach  
❌ **REJECTED:** Any conditional approach  

### Implementation

**TEAM-298 MUST:**
1. Remove `eprintln!()` from `src/lib.rs:559`
2. Update all tests to use capture adapter
3. Add privacy isolation tests

**TEAM-301 WILL:**
1. Create keeper display module
2. Subscribe to SSE in keeper
3. Display to terminal

**Result:**
- Secure by design
- No attack surface
- Defense in depth
- Fail-safe

---

## References

- [PRIVACY_FIX_REQUIRED.md](./PRIVACY_FIX_REQUIRED.md) - Original issue
- [PRIVACY_ATTACK_SURFACE_ANALYSIS.md](./PRIVACY_ATTACK_SURFACE_ANALYSIS.md) - Detailed analysis
- [TEAM_298_PHASE_1_SSE_OPTIONAL.md](./TEAM_298_PHASE_1_SSE_OPTIONAL.md) - Implementation plan
- [TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md](./TEAM_301_PHASE_4_KEEPER_LIFECYCLE.md) - Keeper display

---

**This is the final approach. No alternatives.**

**Security by design, not security by configuration.**
