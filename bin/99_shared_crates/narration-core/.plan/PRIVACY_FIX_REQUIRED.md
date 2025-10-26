# 🚨 CRITICAL: Privacy Violation in Current Implementation

**Discovered By:** TEAM-298 Review  
**Severity:** CRITICAL  
**Status:** MUST FIX IMMEDIATELY  
**Affects:** Phases 0 & 1 (TEAM-297, TEAM-298)

---

## The Problem

### Current Implementation (BROKEN)

```rust
// In narrate_at_level() - src/lib.rs:559
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
```

**This emits ALL narration to global stderr!**

### Privacy Violations

1. **Multi-tenant data leak:**
   - User A submits inference job
   - User B submits different job
   - **Both see each other's narration on stderr!**

2. **Sensitive data exposure:**
   - job_id visible globally
   - correlation_id visible globally
   - Inference prompts/responses visible globally
   - Model names, worker IDs exposed

3. **No isolation:**
   - All jobs share same stderr
   - No way to separate output
   - Security nightmare in production

### Example Attack Scenario

```
Terminal 1 (User A):
$ rbee-keeper infer --prompt "My secret API key is sk-..."
[queen     ] infer_start    : Starting inference

Terminal 2 (User B):
# User B sees User A's narration on SAME stderr!
[queen     ] infer_start    : Starting inference
[worker    ] load_model     : Loading model for job-abc123
[worker    ] inference      : Processing prompt: "My secret API key is sk-..."
                              ^^^^^ USER B SEES USER A'S SECRET!
```

---

## The Fix

### Architecture Change

**OLD (BROKEN):**
```
narrate_at_level()
  ├─ stderr: ALWAYS (global) ❌ PRIVACY VIOLATION
  ├─ SSE: if channel exists
  └─ tracing: optional
```

**NEW (SECURE):**
```
narrate_at_level()
  ├─ stderr: NEVER (removed for security) ✅
  ├─ SSE: if job_id + channel exists ✅
  ├─ tracing: optional
  └─ Keeper mode: can print (single-user only) ✅
```

### Implementation Changes

#### 1. Remove Global stderr Output

**File:** `src/lib.rs` - `narrate_at_level()`

```rust
// TEAM-298: REMOVE THIS (privacy violation)
// eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);

// TEAM-298: NEW - Only emit to SSE (job-scoped)
if sse_sink::is_enabled() {
    let _sse_sent = sse_sink::try_send(&fields);
    // Narration ONLY goes to SSE channel (isolated per job)
}
```

#### 2. Add Keeper Display Mode

**File:** `src/lib.rs`

```rust
// TEAM-298: Add environment variable for keeper mode
static KEEPER_MODE: once_cell::sync::Lazy<bool> = once_cell::sync::Lazy::new(|| {
    std::env::var("RBEE_KEEPER_MODE")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
});

pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // ...
    
    // TEAM-298: Only print to stderr in keeper mode (single-user)
    if *KEEPER_MODE {
        eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
    }
    
    // TEAM-298: Always try SSE (job-scoped, secure)
    if sse_sink::is_enabled() {
        let _sse_sent = sse_sink::try_send(&fields);
    }
    
    // ...
}
```

#### 3. Update SSE to be Primary

**Philosophy change:**
- **Before:** stderr primary, SSE bonus
- **After:** SSE primary, stderr only in keeper mode

```rust
// Multi-user daemons (queen, hive, worker):
// - NO stderr output (security)
// - SSE only (job-scoped)

// Single-user CLI (keeper):
// - Can print to stderr (RBEE_KEEPER_MODE=1)
// - User's own terminal, no privacy issue
```

---

## Migration Path

### Phase 1 (TEAM-298) - IMMEDIATE

1. **Add keeper mode flag**
   - Environment variable: `RBEE_KEEPER_MODE`
   - Default: `false` (secure)

2. **Conditional stderr output**
   - Only if keeper mode enabled
   - Multi-user daemons never print

3. **Update tests**
   - Tests can enable keeper mode
   - Or use capture adapter (already works)

### Phase 4 (TEAM-301) - Keeper Integration

1. **Keeper sets mode**
   ```rust
   // In keeper main()
   std::env::set_var("RBEE_KEEPER_MODE", "1");
   ```

2. **Keeper displays output**
   - Keeper is single-user CLI
   - Safe to print to terminal
   - User sees their own jobs only

---

## Security Model

### Multi-User Daemons (queen, hive, worker)

```rust
// NEVER print to stderr (multi-tenant)
RBEE_KEEPER_MODE=0  // Default

// Narration flow:
n!("action", "msg")
  → Has job_id? → SSE channel (isolated)
  → No job_id? → DROPPED (security)
  → stderr? → NEVER (privacy)
```

### Single-User CLI (keeper)

```rust
// CAN print to stderr (single-user)
RBEE_KEEPER_MODE=1  // Set by keeper

// Narration flow:
n!("action", "msg")
  → Has job_id? → SSE channel (if available)
  → stderr? → YES (user's own terminal)
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_no_stderr_in_multi_user_mode() {
    // Default mode (multi-user)
    std::env::remove_var("RBEE_KEEPER_MODE");
    
    let adapter = CaptureAdapter::install();
    n!("test", "Should not print to stderr");
    
    // Verify no stderr output (use capture adapter instead)
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}

#[test]
fn test_stderr_in_keeper_mode() {
    // Keeper mode (single-user)
    std::env::set_var("RBEE_KEEPER_MODE", "1");
    
    n!("test", "Can print to stderr");
    
    // Visual verification in terminal (keeper mode)
    std::env::remove_var("RBEE_KEEPER_MODE");
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_multi_tenant_isolation() {
    // Simulate two users
    let job_a = "user-a-job";
    let job_b = "user-b-job";
    
    // Create separate SSE channels
    sse_sink::create_job_channel(job_a.to_string(), 100);
    sse_sink::create_job_channel(job_b.to_string(), 100);
    
    let rx_a = sse_sink::take_job_receiver(job_a).unwrap();
    let rx_b = sse_sink::take_job_receiver(job_b).unwrap();
    
    // User A narration
    let ctx_a = NarrationContext::new().with_job_id(job_a);
    with_narration_context(ctx_a, async {
        n!("secret_a", "User A's secret data");
    }).await;
    
    // User B narration
    let ctx_b = NarrationContext::new().with_job_id(job_b);
    with_narration_context(ctx_b, async {
        n!("secret_b", "User B's secret data");
    }).await;
    
    // Verify isolation
    let event_a = rx_a.recv().await.unwrap();
    let event_b = rx_b.recv().await.unwrap();
    
    assert_eq!(event_a.human, "User A's secret data");
    assert_eq!(event_b.human, "User B's secret data");
    
    // CRITICAL: User A never sees User B's data!
    // CRITICAL: User B never sees User A's data!
}
```

---

## Rollout Plan

### Step 1: Add Keeper Mode (TEAM-298)
- Add `RBEE_KEEPER_MODE` environment variable
- Make stderr conditional
- Default to secure (no stderr)

### Step 2: Update Tests (TEAM-298)
- Enable keeper mode in tests
- Or use capture adapter
- Verify no stderr in multi-user mode

### Step 3: Update Daemons (TEAM-299)
- Ensure queen/hive/worker never set keeper mode
- Verify SSE-only operation
- Test multi-tenant isolation

### Step 4: Update Keeper (TEAM-301)
- Set `RBEE_KEEPER_MODE=1` in keeper
- Display output to user terminal
- Single-user, safe to print

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

## Risk Assessment

### HIGH RISK if not fixed:
- ❌ Privacy violations in production
- ❌ Sensitive data exposure
- ❌ Compliance issues (GDPR, etc.)
- ❌ Security audit failures

### LOW RISK with fix:
- ✅ Job-scoped narration (SSE)
- ✅ No cross-job leaks
- ✅ Keeper mode for single-user
- ✅ Secure by default

---

## Action Items

### TEAM-298 (IMMEDIATE):
- [ ] Add `RBEE_KEEPER_MODE` environment variable
- [ ] Make stderr conditional on keeper mode
- [ ] Update `narrate_at_level()` to remove global stderr
- [ ] Add multi-tenant isolation tests
- [ ] Update all existing tests

### TEAM-299:
- [ ] Verify daemons never set keeper mode
- [ ] Test SSE-only operation
- [ ] Document security model

### TEAM-301:
- [ ] Set keeper mode in keeper
- [ ] Display output to terminal
- [ ] Test single-user experience

---

## Conclusion

**Current implementation has CRITICAL privacy violation.**

**Must fix IMMEDIATELY in Phase 1 (TEAM-298).**

**Solution: Remove global stderr, use SSE + keeper mode.**

**This is NOT optional - it's a security requirement!**
