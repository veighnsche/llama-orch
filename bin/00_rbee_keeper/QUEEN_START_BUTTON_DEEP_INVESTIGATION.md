# Deep Investigation: Queen Start Button Implementation

**Date:** Oct 28, 2025  
**Status:** 🔴 CRITICAL BUGS FOUND  
**Investigator:** TEAM-335

---

## Executive Summary

**CRITICAL:** The "Queen Start" button in the UI **WILL NOT WORK** - all Tauri commands were removed by TEAM-334 but the UI still calls them.

**Secondary Issue:** The `#[with_timeout]` and `#[with_job_id]` macros fundamentally conflict with Tauri's command system.

---

## 1. Current Architecture Flow

### UI → Backend Wiring

```typescript
// ServicesPage.tsx:28
await invoke("queen_start");
```

↓

```rust
// main.rs:84-87 - MISSING HANDLER!
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        ssh_list,  // ← Only this exists!
        // queen_start missing!
    ])
```

↓

```rust
// handlers/queen.rs:56-68 - Business logic exists but unreachable!
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    match action {
        QueenAction::Start => {
            let config = StartConfig { /* ... */ };
            start_daemon(config).await?;
            Ok(())
        }
    }
}
```

---

## 2. The TEAM-334 Massacre

**What Happened:**
TEAM-334 removed **ALL** queen commands from `tauri_commands.rs`:
- ❌ `queen_start` - DELETED
- ❌ `queen_stop` - DELETED
- ❌ `queen_install` - DELETED
- ❌ `queen_rebuild` - DELETED
- ❌ `queen_uninstall` - DELETED

**Why:**
> "Architecture in flux... Better to rebuild when architecture stabilizes"

**Problem:**
The UI was NOT updated to reflect this deletion. All buttons call non-existent commands.

---

## 3. Critical Bug #1: Missing Tauri Command

### Current State

```typescript
// UI calls this (ServicesPage.tsx:28)
await invoke("queen_start");
```

```rust
// But this doesn't exist!
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    // 🔥 DELETED BY TEAM-334
}
```

### What Happens

1. User clicks Play button
2. `handleCommand("queen-start")` fires
3. `invoke("queen_start")` called
4. **Tauri error: "Command queen_start not found"**
5. Catch block: `console.error("Command failed:", error)`
6. User sees nothing (error only in console)
7. Button becomes enabled again (no visual feedback)

### User Experience

❌ **Silent failure** - Button click does nothing  
❌ **No error message** - User confused  
❌ **No loading state** - Button just blinks  

---

## 4. Critical Bug #2: Macro Incompatibility

### The Timeout Macro Problem

```rust
// daemon-lifecycle/src/start.rs:214
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    // ...
}
```

**What these macros do:**

1. **`#[with_timeout]`** - Wraps function body in:
   ```rust
   TimeoutEnforcer::new(Duration::from_secs(120))
       .with_label("Start daemon")
       .enforce(__start_daemon_inner())
       .await
   ```

2. **`#[with_job_id]`** - Wraps function body in:
   ```rust
   let __ctx = start_config.job_id.as_ref()
       .map(|jid| NarrationContext::new().with_job_id(jid));
   
   if let Some(__ctx) = __ctx {
       with_narration_context(__ctx, __impl).await
   } else {
       __impl.await
   }
   ```

### Why This Breaks Tauri

**Problem 1: Double Async Wrapping**

```rust
// Tauri expects this signature:
#[tauri::command]
pub async fn queen_start() -> Result<String, String>

// But we need to call:
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120)]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32>
```

The macros create nested async blocks that don't compose well with Tauri's command macro.

**Problem 2: Return Type Mismatch**

- Tauri commands: `Result<String, String>` (serializable)
- Business logic: `Result<u32, anyhow::Error>` (PID with rich errors)

**Problem 3: Parameter Mismatch**

- Tauri commands: Flat parameters (no config structs)
- Business logic: `StartConfig` struct (contains SSH config, daemon config, job_id)

**Problem 4: Job ID Mystery**

```rust
// Where does job_id come from in Tauri?
pub struct StartConfig {
    pub job_id: Option<String>,  // ← Tauri has no job system!
}
```

Tauri is **client-side** - it doesn't have queen-rbee's job system. The `job_id` field is meaningless in this context.

---

## 5. Edge Cases & Potential Bugs

### Edge Case 1: Timeout in GUI Context

```rust
// This narration fires after 120 seconds:
n!("timeout_warning", "⏱️  Operation is taking longer than expected (120s)...");
```

**Problem:** Where does this narration go?
- CLI: stderr ✅
- Tauri: **NOWHERE** ❌

There's no SSE stream in Tauri. Narration goes to stdout/stderr, but Tauri doesn't capture that.

### Edge Case 2: Process Already Running

```rust
// start_daemon checks if daemon is running via health endpoint
let is_running = check_daemon_health(&health_url).await;
```

**Scenario:**
1. User clicks Start
2. Queen already running
3. `start_daemon()` returns error
4. Error message lost in conversion to String
5. User sees: "Command failed: Operation failed"

**Better UX:**
Should detect "already running" and show specific message.

### Edge Case 3: Binary Not Found

```rust
// start_daemon looks for binary in:
// 1. ~/.local/bin/queen-rbee
// 2. target/release/queen-rbee
// 3. target/debug/queen-rbee

if binary_path == "NOT_FOUND" {
    anyhow::bail!("Binary not found. Install it first with install_daemon()");
}
```

**Problem:** User clicks Start before Install
**Result:** Error message: "Binary not found. Install it first with install_daemon()"
**UX Issue:** Mentions `install_daemon()` which is a Rust function, not a user action

### Edge Case 4: SSH vs Localhost Confusion

```rust
// handlers/queen.rs:62
let config = StartConfig {
    ssh_config: SshConfig::localhost(),  // ← Always localhost
    daemon_config,
    job_id: None,  // ← Always None in Tauri
};
```

**Hard-coded assumptions:**
- Queen is always on localhost
- No remote operations
- No SSH keys needed

**But what if:**
- User runs Tauri app on machine A
- Wants to start queen on machine B
- Current code will try to start on machine A (wrong!)

### Edge Case 5: Multiple Tauri Windows

**Scenario:**
1. User opens 2 Tauri windows
2. Both windows show Queen as "stopped"
3. User clicks Start in window 1
4. Queen starts successfully
5. Window 2 still shows "stopped" (no refresh)

**No state synchronization between windows!**

### Edge Case 6: Port Conflicts

```rust
// handlers/queen.rs:49
let port: u16 = queen_url
    .split(':')
    .next_back()
    .and_then(|p| p.parse().ok())
    .unwrap_or(7833);  // ← Default port
```

**Scenario:**
1. Port 7833 already in use (by something else)
2. `start_daemon` spawns queen with `--port 7833`
3. Queen fails to bind (silent failure in background)
4. Health check fails after 30 attempts
5. Timeout after 2 minutes
6. Error: "Daemon started but failed health check"

**User has no idea port is the issue!**

### Edge Case 7: Permission Denied

**Scenario:**
1. User clicks Start
2. Binary at `~/.local/bin/queen-rbee` not executable
3. `nohup ~/.local/bin/queen-rbee --port 7833 > /dev/null 2>&1 &` fails
4. SSH returns error code
5. Error message: "Failed to start daemon"

**Missing detail:** Needs `chmod +x` fix suggestion

### Edge Case 8: Zombie Processes

```rust
// start_daemon spawns with nohup
let start_cmd = format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args);
```

**Problem:** If daemon crashes during startup:
1. PID returned successfully
2. Health check fails
3. Error thrown
4. Process remains in process table (zombie)
5. User clicks Start again → "Address already in use"

**No cleanup mechanism!**

### Edge Case 9: Config File Race Condition

**Scenario:**
1. User clicks Start in window 1
2. Queen loads `~/.llorch.toml`
3. User edits config file in editor
4. User clicks Start in window 2
5. Window 2 loads different config
6. Two queens with different configs fighting

**No file locking!**

### Edge Case 10: Network Timeout on Health Check

```rust
// start.rs:277
let poll_config = HealthPollConfig::new(&daemon_config.health_url)
    .with_max_attempts(30);
```

**30 attempts × exponential backoff = ~30 seconds**

But wrapped in `#[with_timeout(secs = 120)]`

**If network slow:**
- Health check might take 60 seconds
- User sees nothing for 60 seconds
- Then suddenly "success"

**No progress indicator!**

---

## 6. The Macro Expansion Catastrophe

### What `#[with_timeout]` Actually Does

```rust
// Original function
#[with_timeout(secs = 120, label = "Start daemon")]
pub async fn start_daemon(config: StartConfig) -> Result<u32> {
    // ... actual logic ...
}

// Expands to:
pub async fn start_daemon(config: StartConfig) -> Result<u32> {
    async fn __start_daemon_inner(config: StartConfig) -> Result<u32> {
        // ... actual logic ...
    }
    
    TimeoutEnforcer::new(Duration::from_secs(120))
        .with_label("Start daemon")
        .enforce(__start_daemon_inner(config))
        .await
}
```

### What `#[with_job_id]` Actually Does

```rust
// Stacked macros
#[with_job_id(config_param = "start_config")]
#[with_timeout(secs = 120)]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    // ... logic ...
}

// First expansion (with_timeout):
#[with_job_id(config_param = "start_config")]
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    async fn __start_daemon_inner(start_config: StartConfig) -> Result<u32> {
        // ... logic ...
    }
    TimeoutEnforcer::new(Duration::from_secs(120))
        .enforce(__start_daemon_inner(start_config))
        .await
}

// Second expansion (with_job_id):
pub async fn start_daemon(start_config: StartConfig) -> Result<u32> {
    let __ctx = start_config.job_id.as_ref()
        .map(|jid| NarrationContext::new().with_job_id(jid));
    
    let __impl = async move {
        async fn __start_daemon_inner(start_config: StartConfig) -> Result<u32> {
            // ... logic ...
        }
        TimeoutEnforcer::new(Duration::from_secs(120))
            .enforce(__start_daemon_inner(start_config))
            .await
    };
    
    if let Some(__ctx) = __ctx {
        with_narration_context(__ctx, __impl).await
    } else {
        __impl.await
    }
}
```

### Why This is CATASTROPHIC for Tauri

**Problem 1: Variable Capture**

```rust
// The macro does this:
let __impl = async move {
    // ... uses start_config ...
};

// But Tauri needs:
#[tauri::command]
pub async fn queen_start() -> Result<String, String> {
    // How do we pass start_config here???
}
```

**No way to pass config struct through Tauri's parameter system!**

**Problem 2: Context Propagation**

```rust
// The macro expects:
with_narration_context(__ctx, __impl).await

// But Tauri is stateless!
// No thread-local storage across await points
// No way to propagate context
```

**Problem 3: Error Conversion Hell**

```rust
// Business logic returns:
Result<u32, anyhow::Error>

// Wrapped in TimeoutEnforcer:
Result<u32, anyhow::Error>  // Same

// Wrapped in NarrationContext:
Result<u32, anyhow::Error>  // Still same

// But Tauri needs:
Result<String, String>

// How do we convert?
// - u32 → String: Easy (format!("{}", pid))
// - anyhow::Error → String: Lossy (loses context chain)
```

**Problem 4: Async Stack Explosion**

```rust
// Macro creates 3 async layers:
1. pub async fn start_daemon()          // Tauri entry
2.   let __impl = async move { ... }    // with_job_id wrapper
3.     TimeoutEnforcer::enforce(async { ... })  // timeout wrapper
4.       async fn __start_daemon_inner() { ... }  // actual logic

// Total: 4 async layers!
// Each layer adds overhead
// Each layer is a potential panic boundary
```

---

## 7. Correct Implementation Strategy

### Option A: Thin Tauri Wrapper (RECOMMENDED)

```rust
// tauri_commands.rs
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    // THIN WRAPPER - No macros, no magic
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    
    // 1. Load config
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    
    // 2. Call handler (which has all the macros)
    handle_queen(QueenAction::Start, &queen_url)
        .await
        .map(|_| "Queen started successfully".to_string())
        .map_err(|e| format!("Start failed: {}", e))
}
```

**Pros:**
- ✅ Clean separation: Tauri layer vs business logic
- ✅ No macro conflicts
- ✅ Error messages preserved
- ✅ Easy to test

**Cons:**
- ❌ No job_id (can't use SSE in Tauri anyway)
- ❌ No progress updates during 2-minute timeout
- ❌ User sees nothing until completion

### Option B: Event-Based Progress (BETTER UX)

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_start(window: tauri::Window) -> Result<String, String> {
    // Custom narration sink that emits Tauri events
    struct TauriNarrationSink {
        window: tauri::Window,
    }
    
    impl NarrationSink for TauriNarrationSink {
        fn emit(&self, event: NarrationEvent) {
            // Emit Tauri event to frontend
            self.window.emit("narration", event).ok();
        }
    }
    
    // Install custom sink
    set_narration_sink(TauriNarrationSink { window: window.clone() });
    
    // Now call business logic (narration goes to frontend!)
    handle_queen(QueenAction::Start, &queen_url).await?;
    
    Ok("Started".to_string())
}
```

```typescript
// ServicesPage.tsx
import { listen } from "@tauri-apps/api/event";

useEffect(() => {
  const unlisten = listen("narration", (event) => {
    const narration = event.payload as NarrationEvent;
    
    if (narration.action === "timeout_warning") {
      showToast("Operation is taking longer than expected...");
    }
    
    if (narration.action === "start_complete") {
      showToast("Queen started successfully!");
    }
  });
  
  return () => { unlisten.then(f => f()); };
}, []);
```

**Pros:**
- ✅ Real-time progress updates
- ✅ Timeout warnings visible to user
- ✅ Better UX

**Cons:**
- ❌ More complex
- ❌ Requires Tauri event system
- ❌ Narration system needs to support multiple sinks

### Option C: Polling-Based Status (SIMPLEST)

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    // Just start it, don't wait
    tokio::spawn(async {
        handle_queen(QueenAction::Start, &queen_url).await.ok();
    });
    
    Ok("Starting...".to_string())
}

#[tauri::command]
#[specta::specta]
pub async fn queen_status() -> Result<String, String> {
    // Let UI poll this every 2 seconds
    use daemon_lifecycle::check_daemon_health;
    
    let is_running = check_daemon_health("http://localhost:7833/health").await;
    
    Ok(if is_running { "healthy" } else { "stopped" }.to_string())
}
```

```typescript
// ServicesPage.tsx
const handleCommand = async (command: string) => {
  if (command === "queen-start") {
    await invoke("queen_start");
    
    // Poll status until healthy
    const interval = setInterval(async () => {
      const status = await invoke("queen_status");
      if (status === "healthy") {
        clearInterval(interval);
        showToast("Queen started!");
      }
    }, 2000);
  }
};
```

**Pros:**
- ✅ Simplest implementation
- ✅ No macro conflicts
- ✅ Works with existing architecture

**Cons:**
- ❌ Polling overhead
- ❌ No timeout visibility
- ❌ Fire-and-forget (errors lost)

---

## 8. Recommended Implementation

### Step 1: Add Tauri Command

```rust
// tauri_commands.rs

/// Start queen-rbee daemon on localhost
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    use observability_narration_core::n;
    
    n!("queen_start", "🚀 Starting queen-rbee from Tauri GUI...");
    
    // Load config
    let config = Config::load()
        .map_err(|e| format!("Failed to load config: {}", e))?;
    
    let queen_url = config.queen_url();
    
    // Call handler (has all the timeout/job_id macros)
    // Note: job_id will be None in Tauri context (no SSE)
    handle_queen(QueenAction::Start, &queen_url)
        .await
        .map(|_| {
            n!("queen_start", "✅ Queen started successfully");
            "Queen started successfully".to_string()
        })
        .map_err(|e| {
            let err_msg = format!("Failed to start queen: {}", e);
            n!("queen_start", "❌ {}", err_msg);
            err_msg
        })
}
```

### Step 2: Register in main.rs

```rust
// main.rs:84-87
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        ssh_list,
        queen_start,  // ← ADD THIS
    ])
```

### Step 3: Update TypeScript Bindings

```bash
# Generate new bindings
cargo test --lib export_typescript_bindings
```

### Step 4: Add Error Handling in UI

```typescript
// ServicesPage.tsx
const handleCommand = async (command: string) => {
  setActiveCommand(command);
  setIsExecuting(true);

  try {
    switch (command) {
      case "queen-start":
        const result = await invoke<string>("queen_start");
        // Show success toast
        showToast(result, "success");
        // Refresh status
        await refreshStatus();
        break;
      // ...
    }
  } catch (error) {
    // Show error toast with actual message
    showToast(
      error instanceof Error ? error.message : "Command failed",
      "error"
    );
  } finally {
    setIsExecuting(false);
  }
};
```

---

## 9. Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_queen_start_success() {
        // Mock: Queen not running, binary exists
        let result = queen_start().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_queen_start_already_running() {
        // Mock: Queen already running
        let result = queen_start().await;
        // Should return error with helpful message
        assert!(result.unwrap_err().contains("already running"));
    }
    
    #[tokio::test]
    async fn test_queen_start_binary_not_found() {
        // Mock: Binary doesn't exist
        let result = queen_start().await;
        assert!(result.unwrap_err().contains("not found"));
    }
}
```

### Integration Tests

1. **Clean Start:** Queen not running → Start → Health check passes
2. **Already Running:** Queen running → Start → Graceful error
3. **Binary Missing:** No binary → Start → Install suggestion
4. **Port Conflict:** Port in use → Start → Port error message
5. **Timeout:** Slow startup → Start → Timeout after 2min
6. **Concurrent Starts:** Two windows click Start → One succeeds, one fails gracefully

---

## 10. Documentation Updates Needed

### Files to Update

1. **UI Documentation**
   - `bin/00_rbee_keeper/ui/README.md` - Add queen_start command
   - `bin/00_rbee_keeper/TAURI_INTEGRATION.md` - Update command list

2. **API Documentation**
   - `bin/00_rbee_keeper/ui/TAURI_TYPEGEN_SETUP.md` - Add example

3. **Handoff Document**
   - Create `TEAM_335_QUEEN_START_IMPLEMENTATION.md`

---

## 11. Critical Warnings for Implementation

### 🚨 DO NOT

1. **DO NOT** add `#[with_timeout]` to Tauri commands directly
2. **DO NOT** add `#[with_job_id]` to Tauri commands directly
3. **DO NOT** return `Result<u32, anyhow::Error>` from Tauri commands
4. **DO NOT** pass complex structs as Tauri parameters
5. **DO NOT** assume job_id exists in Tauri context

### ✅ DO

1. **DO** keep Tauri commands thin wrappers
2. **DO** convert all errors to String
3. **DO** use existing handlers for business logic
4. **DO** test error cases thoroughly
5. **DO** add narration for debugging

---

## 12. Summary

### Current State
❌ **BROKEN** - UI calls non-existent Tauri command

### Root Causes
1. TEAM-334 deleted all commands without updating UI
2. Macro-decorated functions incompatible with Tauri
3. No job_id in Tauri context (SSE doesn't make sense)
4. No progress updates during 2-minute timeout

### Recommended Fix
**Option A** (Thin Wrapper) - 30 minutes work
- Simple, clean, works
- Trade-off: No progress updates

### Alternative Fix
**Option B** (Event-Based) - 4 hours work
- Better UX, real-time updates
- Trade-off: More complex

### Edge Cases Found
10 major edge cases documented (see Section 5)

### Estimated Effort
- Option A: 30 minutes
- Option B: 4 hours
- Testing: 2 hours
- Documentation: 1 hour

**Total: 3-7 hours depending on approach**

---

**END OF INVESTIGATION**
