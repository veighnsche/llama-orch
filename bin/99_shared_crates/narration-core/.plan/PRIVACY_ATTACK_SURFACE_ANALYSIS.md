# Privacy Attack Surface Analysis

**Date:** 2025-10-26  
**Reviewer:** TEAM-298  
**Severity:** CRITICAL

---

## Current Implementation Review

### Attack Surface Found

**File:** `src/lib.rs:559`
```rust
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
```

**This is UNCONDITIONAL and ALWAYS executes!**

### Problems with Proposed "Keeper Mode" Solution

❌ **Runtime check is exploitable:**
```rust
if is_keeper_mode() {  // ← Can be set by attacker!
    eprintln!(...);
}
```

**Attack vectors:**
1. Malicious actor sets `RBEE_KEEPER_MODE=1` in daemon
2. Accidental inheritance from parent process
3. Environment variable injection
4. Tests forget to unset it
5. Configuration error enables it in production

❌ **Code path still exists:**
- The `eprintln!()` code is compiled into daemons
- Just guarded by runtime check
- Exploitable if guard bypassed

❌ **No defense in depth:**
- Single point of failure (env var check)
- No compile-time safety
- No physical separation

---

## Proper Solution: Complete Removal

### Principle: If code doesn't exist, it can't be exploited

**Remove ALL stderr from narration-core completely.**

### Architecture Change

**OLD (BROKEN):**
```
narrate_at_level()
  ├─ eprintln!() ← ATTACK SURFACE
  ├─ SSE
  └─ tracing
```

**NEW (SECURE):**
```
narration-core:
  narrate_at_level()
    ├─ SSE (job-scoped, secure)
    └─ tracing

keeper (separate):
  keeper_display()
    ├─ Subscribe to SSE stream
    └─ Print to terminal
```

### Implementation

#### 1. Remove stderr Completely

**File:** `src/lib.rs`

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return;
    };

    // TEAM-298: Select message based on mode
    let mode = mode::get_narration_mode();
    let message = match mode {
        mode::NarrationMode::Human => &fields.human,
        mode::NarrationMode::Cute => fields.cute.as_ref().unwrap_or(&fields.human),
        mode::NarrationMode::Story => fields.story.as_ref().unwrap_or(&fields.human),
    };

    // TEAM-298: REMOVED - No stderr output in narration-core!
    // This eliminates the attack surface completely.
    // Keeper will display via separate mechanism.

    // TEAM-298: SSE is PRIMARY and ONLY output
    if sse_sink::is_enabled() {
        let _sse_sent = sse_sink::try_send(&fields);
    }

    // Tracing (optional)
    match tracing_level {
        Level::TRACE => emit_event!(Level::TRACE, fields),
        Level::DEBUG => emit_event!(Level::DEBUG, fields),
        Level::INFO => emit_event!(Level::INFO, fields),
        Level::WARN => emit_event!(Level::WARN, fields),
        Level::ERROR => emit_event!(Level::ERROR, fields),
    }

    // Test capture
    #[cfg(any(test, feature = "test-support"))]
    {
        capture::notify(fields);
    }
}
```

#### 2. Keeper Displays via SSE Subscription

**File:** `bin/00_rbee_keeper/src/display.rs` (new)

```rust
use observability_narration_core::sse_sink::{NarrationEvent};
use tokio::sync::mpsc;

/// Display narration events to terminal
///
/// Keeper subscribes to SSE stream and displays events.
/// This is ONLY used in keeper (single-user CLI).
pub async fn display_narration_stream(mut rx: mpsc::Receiver<NarrationEvent>) {
    while let Some(event) = rx.recv().await {
        // Display to terminal (keeper's own terminal, single-user)
        eprintln!("{}", event.formatted);
    }
}
```

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Keeper creates its own SSE channel for display
    let keeper_job_id = "keeper-display";
    sse_sink::create_job_channel(keeper_job_id.to_string(), 1000);
    let rx = sse_sink::take_job_receiver(keeper_job_id).unwrap();
    
    // Spawn display task
    tokio::spawn(async move {
        display_narration_stream(rx).await;
    });
    
    // Set context so keeper's narration goes to its own channel
    let ctx = NarrationContext::new().with_job_id(keeper_job_id);
    
    with_narration_context(ctx, async {
        // Run keeper
        run_keeper().await
    }).await
}
```

#### 3. Tests Use Capture Adapter

**All tests:**
```rust
#[test]
fn test_narration() {
    // Tests NEVER use stderr
    // Always use capture adapter
    let adapter = CaptureAdapter::install();
    
    n!("test", "Test message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
}
```

---

## Security Analysis

### Before (BROKEN)

**Attack Surface:**
- Global eprintln in narration-core
- Runtime guard (bypassable)
- Environment variable (exploitable)
- Code exists in all binaries

**Exploitability:** HIGH
- Set env var → instant data leak
- Single point of failure
- No defense in depth

### After (SECURE)

**Attack Surface:**
- NONE in narration-core
- No stderr code path
- SSE only (job-scoped)
- Keeper display is separate

**Exploitability:** NONE
- No code to exploit in daemons
- Keeper display is isolated
- Physical separation
- Multiple layers of defense

---

## Defense in Depth

### Layer 1: Code Removal
- No eprintln in narration-core
- Code physically doesn't exist in daemons
- Can't be exploited if not compiled in

### Layer 2: SSE Isolation
- Job-scoped channels
- job_id required for routing
- No cross-job leaks
- Fail-fast security

### Layer 3: Keeper Separation
- Keeper explicitly subscribes
- Separate display function
- Only in keeper binary
- Clear separation of concerns

### Layer 4: Testing
- Tests use capture adapter
- No stderr dependency
- Can't accidentally leak
- Compile-time safe

---

## Migration

### Step 1: Remove stderr from narration-core

```diff
- eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
+ // REMOVED - No stderr in narration-core (security)
```

### Step 2: Update all tests

```diff
- // Tests relied on stderr
+ let adapter = CaptureAdapter::install();
+ // Now use capture adapter
```

### Step 3: Add keeper display

```rust
// keeper/src/display.rs
pub async fn display_narration_stream(rx) {
    while let Some(event) = rx.recv().await {
        eprintln!("{}", event.formatted);
    }
}
```

### Step 4: Wire keeper

```rust
// keeper/src/main.rs
let rx = sse_sink::take_job_receiver("keeper-display")?;
tokio::spawn(display_narration_stream(rx));
```

---

## Advantages

### Security
✅ No attack surface in narration-core  
✅ No exploitable code path  
✅ No environment variable to manipulate  
✅ Defense in depth  

### Simplicity
✅ Cleaner separation  
✅ No conditional logic  
✅ No runtime checks  
✅ Easier to reason about  

### Testability
✅ Tests use capture adapter  
✅ No stderr dependency  
✅ Compile-time safe  
✅ No accidental leaks  

### Maintainability
✅ Clear separation of concerns  
✅ Keeper display is explicit  
✅ No magic environment variables  
✅ Self-documenting code  

---

## Rejected Approaches

### ❌ Keeper Mode Environment Variable

**Why rejected:**
- Exploitable (attacker sets env var)
- Accidental inheritance
- Single point of failure
- Code path still exists

### ❌ Feature Flag (`keeper-mode`)

**Why rejected:**
- Still compiled into daemons (optional)
- Could be enabled accidentally
- Cargo features are additive
- Not defense in depth

### ❌ Conditional Compilation (`#[cfg(keeper)]`)

**Why better but still rejected:**
- Requires different build targets
- Complex build system
- Easy to misconfigure
- Separation not clear enough

---

## Conclusion

**The ONLY secure solution is complete removal.**

1. **Remove ALL eprintln from narration-core**
2. **SSE is the ONLY output**
3. **Keeper displays via separate subscription**
4. **Tests use capture adapter**

This eliminates the attack surface completely and provides:
- Physical separation (code doesn't exist)
- Defense in depth (multiple layers)
- Compile-time safety (can't be exploited)
- Clear architecture (separation of concerns)

**Any solution that keeps eprintln in narration-core is INSECURE.**
