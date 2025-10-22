# TEAM-222: SSH Client NARRATION INVENTORY

**Component:** `bin/15_queen_rbee_crates/ssh-client`  
**Date:** Oct 22, 2025  
**Status:** ✅ COMPLETE

---

## Summary

SSH client uses **DEPRECATED** narration pattern (v0.1.0 style).

**CRITICAL:** Uses `Narration::new()` instead of `NarrationFactory` pattern.

---

## 1. Current Implementation (DEPRECATED)

### Narration Pattern (lib.rs:106-134)

```rust
// ❌ DEPRECATED: Old v0.1.0 pattern
use observability_narration_core::Narration;

const ACTOR_SSH_CLIENT: &str = "🔐 ssh-client";
const ACTION_TEST: &str = "test_connection";

// Test starting
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, &target)
    .human(format!("🔐 Testing SSH connection to {}", target))
    .emit();

// Success
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "success")
    .human(format!("✅ SSH connection to {} successful", target))
    .emit();

// Failure
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "failed")
    .human(format!("❌ SSH connection to {} failed: {}", target, error))
    .emit();

// Error
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, "error")
    .human(format!("❌ SSH test error: {}", e))
    .emit();
```

---

## 2. Problems with Current Implementation

### ❌ Deprecated Pattern
- Uses `Narration::new()` (v0.1.0 API)
- Should use `NarrationFactory` (v0.5.0 API)

### ❌ Actor Too Long
- Actor: `"🔐 ssh-client"` (14 chars with emoji)
- Limit: 10 chars
- **Compile error** in v0.5.0

### ❌ No job_id
- Narrations have NO job_id
- Goes to stderr only (not SSE)
- Users can't see SSH test progress in real-time

### ❌ Inconsistent with Other Crates
- hive-lifecycle uses `NarrationFactory` + `.job_id()`
- SSH client uses old `Narration::new()` pattern

---

## 3. Required Migration (v0.5.0)

### Step 1: Update Imports

```rust
// Before (DEPRECATED)
use observability_narration_core::Narration;

// After (v0.5.0)
use observability_narration_core::NarrationFactory;
```

### Step 2: Define Factory

```rust
// Before (DEPRECATED)
const ACTOR_SSH_CLIENT: &str = "🔐 ssh-client";
const ACTION_TEST: &str = "test_connection";

// After (v0.5.0)
const NARRATE: NarrationFactory = NarrationFactory::new("ssh-cli");
//                                                       ^^^^^^^^
//                                                       Max 10 chars, no emoji
```

### Step 3: Update Narration Calls

```rust
// Before (DEPRECATED)
Narration::new(ACTOR_SSH_CLIENT, ACTION_TEST, &target)
    .human(format!("🔐 Testing SSH connection to {}", target))
    .emit();

// After (v0.5.0)
NARRATE
    .action("ssh_test")
    .context(&target)
    .human("🔐 Testing SSH connection to {}")
    .emit();
```

### Step 4: Add job_id Support

SSH client is called from hive-lifecycle's `execute_ssh_test()`, which has access to job_id.

**Update function signature:**

```rust
// Before
pub async fn test_ssh_connection(config: SshConfig) -> Result<SshTestResult>

// After
pub async fn test_ssh_connection(
    config: SshConfig,
    job_id: Option<&str>,  // ← Add job_id parameter
) -> Result<SshTestResult>
```

**Update narrations:**

```rust
// After (with job_id)
let mut narration = NARRATE
    .action("ssh_test")
    .context(&target)
    .human("🔐 Testing SSH connection to {}");

if let Some(job_id) = job_id {
    narration = narration.job_id(job_id);  // ← Include for SSE routing
}

narration.emit();
```

---

## 4. Migration Checklist

- [ ] Update imports: `Narration` → `NarrationFactory`
- [ ] Define factory: `const NARRATE: NarrationFactory = NarrationFactory::new("ssh-cli");`
- [ ] Shorten actor: `"🔐 ssh-client"` → `"ssh-cli"` (7 chars)
- [ ] Update all narration calls: `Narration::new()` → `NARRATE.action()`
- [ ] Add job_id parameter to `test_ssh_connection()`
- [ ] Include `.job_id(job_id)` in all narrations (when present)
- [ ] Update hive-lifecycle's `execute_ssh_test()` to pass job_id
- [ ] Test compilation: `cargo check -p queen-rbee-ssh-client`
- [ ] Test SSE routing: Verify SSH test narrations appear in keeper's SSE stream

---

## 5. Expected Narrations (After Migration)

```rust
// Test starting
NARRATE
    .action("ssh_test")
    .job_id(job_id)
    .context(&target)
    .human("🔐 Testing SSH connection to {}")
    .emit();

// TCP connection
NARRATE
    .action("ssh_tcp")
    .job_id(job_id)
    .context(&target)
    .human("🔌 Establishing TCP connection to {}")
    .emit();

// SSH handshake
NARRATE
    .action("ssh_handshake")
    .job_id(job_id)
    .human("🤝 Performing SSH handshake")
    .emit();

// Authentication
NARRATE
    .action("ssh_auth")
    .job_id(job_id)
    .human("🔑 Authenticating with SSH agent")
    .emit();

// Test command
NARRATE
    .action("ssh_cmd")
    .job_id(job_id)
    .context("echo test")
    .human("📤 Running test command: {}")
    .emit();

// Success
NARRATE
    .action("ssh_success")
    .job_id(job_id)
    .context(&target)
    .human("✅ SSH connection to {} successful")
    .emit();

// Failure
NARRATE
    .action("ssh_failed")
    .job_id(job_id)
    .context(&target)
    .context(&error)
    .human("❌ SSH connection to {0} failed: {1}")
    .emit();
```

---

## 6. Findings

### ❌ Critical Issues
1. **Uses deprecated v0.1.0 API** - `Narration::new()` instead of `NarrationFactory`
2. **Actor too long** - 14 chars (with emoji), limit is 10 chars
3. **No job_id** - Narrations go to stderr, not SSE
4. **Inconsistent with other crates** - hive-lifecycle uses v0.5.0 pattern

### 📋 Recommendations
1. **HIGH PRIORITY:** Migrate to v0.5.0 API (NarrationFactory pattern)
2. **HIGH PRIORITY:** Add job_id support for SSE routing
3. **MEDIUM PRIORITY:** Add more granular narrations (TCP, handshake, auth, command)
4. **LOW PRIORITY:** Consider adding correlation_id support

---

## 7. Code Signatures

```rust
// TEAM-222: Investigated - DEPRECATED v0.1.0 pattern, needs migration to v0.5.0
```

**Files investigated:**
- `bin/15_queen_rbee_crates/ssh-client/src/lib.rs` (264 lines)

---

**TEAM-222 COMPLETE** ✅

**CRITICAL FINDING:** SSH client uses DEPRECATED v0.1.0 narration pattern. Must migrate to v0.5.0 (NarrationFactory + job_id) for consistency and SSE routing.
