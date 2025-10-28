# TEAM-335: Queen Start Button Implementation

**Date:** Oct 28, 2025  
**Status:** ✅ COMPLETE

---

## Mission

Fix the broken Queen Start button in the Tauri GUI by adding thin Tauri command wrappers around existing business logic.

---

## Problem

After TEAM-334 cleanup, the UI still called `invoke("queen_start")` but all Tauri commands were deleted:

```typescript
// ServicesPage.tsx:28 - UI calls this
await invoke("queen_start");

// main.rs:84-87 - But handler doesn't exist!
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        ssh_list,  // ← Only this existed
    ])
```

**Result:** Button click fails silently with "Command not found"

---

## Key Insight

The **business logic already works perfectly** via CLI:

```bash
./rbee queen start    ✅ Works
./rbee queen stop     ✅ Works
./rbee queen install  ✅ Works
```

This proves:
- ✅ `handle_queen()` works
- ✅ `start_daemon()` works  
- ✅ All macros (`#[with_timeout]`, `#[with_job_id]`) work
- ✅ Error handling works

**Only problem:** Missing Tauri bridge layer

---

## Solution: Thin Wrapper Pattern

Added 5 Tauri commands that delegate to existing handlers:

```rust
#[tauri::command]
#[specta::specta]
pub async fn queen_start() -> Result<String, String> {
    use crate::handlers::handle_queen;
    use crate::cli::QueenAction;
    use crate::Config;
    
    let config = Config::load()
        .map_err(|e| format!("Config error: {}", e))?;
    let queen_url = config.queen_url();
    
    handle_queen(QueenAction::Start, &queen_url)
        .await
        .map(|_| "Queen started successfully".to_string())
        .map_err(|e| format!("{}", e))
}
```

**Pattern:**
1. Load config
2. Call existing handler
3. Convert `Result<(), anyhow::Error>` → `Result<String, String>`
4. Return success/error message

---

## Implementation Details

### 1. Added Commands (tauri_commands.rs)

```rust
// TEAM-335: Queen lifecycle commands
- queen_start()       // Start daemon
- queen_stop()        // Stop daemon
- queen_install()     // Install binary (optional path)
- queen_rebuild()     // Rebuild from source
- queen_uninstall()   // Uninstall binary
```

**Lines Added:** ~100 LOC (5 commands × ~20 LOC each)

### 2. Registered in main.rs

```rust
tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![
        ssh_list,
        queen_start,      // ← NEW
        queen_stop,       // ← NEW
        queen_install,    // ← NEW
        queen_rebuild,    // ← NEW
        queen_uninstall,  // ← NEW
    ])
```

### 3. Updated TypeScript Binding Test

```rust
let builder = Builder::<tauri::Wry>::new()
    .commands(collect_commands![
        ssh_list,
        queen_start,
        queen_stop,
        queen_install,
        queen_rebuild,
        queen_uninstall,
    ]);
```

### 4. Generated TypeScript Bindings

```bash
cargo test --package rbee-keeper --lib export_typescript_bindings
```

**Result:** `ui/src/generated/bindings.ts` updated with:

```typescript
export const commands = {
  async queenStart() : Promise<Result<string, string>>,
  async queenStop() : Promise<Result<string, string>>,
  async queenInstall(binary: string | null) : Promise<Result<string, string>>,
  async queenRebuild(withLocalHive: boolean) : Promise<Result<string, string>>,
  async queenUninstall() : Promise<Result<string, string>>,
}
```

---

## Why This Works

### No Macro Conflicts

The thin wrapper sits **above** the macro-decorated functions:

```
UI Layer:          invoke("queen_start")
                        ↓
Tauri Layer:       queen_start() [thin wrapper, no macros]
                        ↓
Handler Layer:     handle_queen(QueenAction::Start)
                        ↓
Business Logic:    start_daemon() [has #[with_timeout], #[with_job_id]]
```

**Separation of concerns:**
- Tauri layer: Simple parameter passing and error conversion
- Handler layer: CLI argument parsing
- Business logic: All the real work (timeouts, SSH, health checks)

### Job ID Handling

In CLI context:
```rust
// job_id is None (no SSE needed)
let config = StartConfig {
    ssh_config: SshConfig::localhost(),
    daemon_config,
    job_id: None,  // ← No SSE in client-side CLI
};
```

The `#[with_job_id]` macro handles this gracefully:
```rust
// If job_id is None, executes directly without context wrapping
if let Some(__ctx) = __ctx {
    with_narration_context(__ctx, __impl).await
} else {
    __impl.await  // ← This path taken for CLI/Tauri
}
```

**Result:** Narration goes to stdout/stderr (captured by Tauri logs)

---

## Files Changed

| File | Change | LOC |
|------|--------|-----|
| `src/tauri_commands.rs` | Added 5 queen commands | +100 |
| `src/main.rs` | Registered commands in handler | +5 |
| `ui/src/generated/bindings.ts` | TypeScript bindings (generated) | +75 |

**Total:** ~180 LOC added

---

## Testing

### Compilation

```bash
✅ cargo check -p rbee-keeper --lib
✅ cargo build -p rbee-keeper --bin rbee-keeper
✅ cargo test --lib export_typescript_bindings
```

### Manual Testing

**Before:**
1. Click Queen Start button
2. Nothing happens (silent failure)
3. Console shows: "Command queen_start not found"

**After:**
1. Click Queen Start button
2. Queen starts (2-minute timeout enforced)
3. Success message: "Queen started successfully"
4. Error cases: Proper error messages shown

### CLI Still Works

```bash
✅ ./rbee queen start
✅ ./rbee queen stop
✅ ./rbee queen install
✅ ./rbee queen rebuild
✅ ./rbee queen uninstall
```

---

## Edge Cases Handled

### 1. Config Loading Failure
```rust
Config::load().map_err(|e| format!("Config error: {}", e))?;
```
**Result:** User sees "Config error: [details]"

### 2. Queen Already Running
```rust
handle_queen(QueenAction::Start, &queen_url).await
```
**Result:** Business logic detects health check passes, returns appropriate error

### 3. Binary Not Found
```rust
handle_queen(QueenAction::Start, &queen_url).await
```
**Result:** `start_daemon()` checks binary locations, returns "Binary not found" error

### 4. Timeout After 2 Minutes
```rust
#[with_timeout(secs = 120, label = "Start daemon")]
```
**Result:** TimeoutEnforcer fires, error propagates to UI

### 5. Port Already in Use
**Result:** Daemon fails to bind, health check fails, error propagates

---

## What We Didn't Do (Good!)

### ❌ NOT Done: Duplicate Business Logic

We **didn't** copy-paste the business logic into Tauri commands. That would create:
- Code duplication (2 implementations of same thing)
- Maintenance burden (fix bugs in 2 places)
- Drift risk (implementations diverge over time)

### ❌ NOT Done: Add Macros to Tauri Layer

We **didn't** add `#[with_timeout]` or `#[with_job_id]` to Tauri commands. That would cause:
- Macro composition conflicts
- Variable capture issues
- Nested async wrapper complexity

### ❌ NOT Done: Custom SSE Implementation

We **didn't** add custom event streaming for progress updates. That would require:
- Tauri event system integration
- Custom narration sink
- Complex state management
- 4+ hours of work

**Trade-off accepted:** User doesn't see progress during 2-minute timeout, but button works correctly.

---

## Future Enhancements

### Option 1: Event-Based Progress (If Needed)

```rust
#[tauri::command]
pub async fn queen_start(window: tauri::Window) -> Result<String, String> {
    // Custom narration sink that emits Tauri events
    window.emit("narration", event).ok();
}
```

```typescript
listen("narration", (event) => {
  if (event.payload.action === "timeout_warning") {
    showToast("Operation taking longer than expected...");
  }
});
```

**Effort:** 4 hours  
**Benefit:** Real-time progress updates

### Option 2: Polling-Based Status

```typescript
const interval = setInterval(async () => {
  const status = await invoke("queen_status");
  if (status === "healthy") {
    clearInterval(interval);
    showToast("Queen started!");
  }
}, 2000);
```

**Effort:** 30 minutes  
**Benefit:** Non-blocking status updates

---

## Architecture Principles Followed

### 1. Thin Wrapper Pattern ✅

Tauri commands are **thin adapters** that:
- Load config
- Call existing handlers
- Convert types
- Return simple strings

**No business logic in Tauri layer!**

### 2. Single Source of Truth ✅

Business logic lives in **one place**:
- CLI uses: `handle_queen()`
- GUI uses: `handle_queen()` (via Tauri wrapper)

**One implementation, two entry points!**

### 3. Type Boundary ✅

Conversion happens at the boundary:
```rust
Result<(), anyhow::Error> → Result<String, String>
```

**Rich errors inside, simple strings at boundary!**

### 4. No Premature Optimization ✅

We **didn't** add:
- Progress streaming (not needed yet)
- Complex error handling (simple messages work)
- State management (stateless commands work)

**YAGNI: You Aren't Gonna Need It!**

---

## Verification Checklist

- [x] All 5 commands compile
- [x] TypeScript bindings generated
- [x] No macro conflicts
- [x] Error messages preserved
- [x] CLI still works (unchanged)
- [x] GUI buttons now work
- [x] No code duplication
- [x] No breaking changes

---

## Documentation Updates

### Updated Files
- ✅ `QUEEN_START_BUTTON_DEEP_INVESTIGATION.md` - Complete analysis
- ✅ `TEAM_335_QUEEN_START_IMPLEMENTATION.md` - This document

### No Updates Needed
- ❌ `TAURI_INTEGRATION.md` - Already documented pattern
- ❌ `TAURI_TYPEGEN_SETUP.md` - Process unchanged

---

## Summary

**Problem:** TEAM-334 deleted Tauri commands, breaking GUI buttons  
**Root Cause:** Missing bridge layer between UI and business logic  
**Solution:** Added thin Tauri wrappers (100 LOC)  
**Result:** GUI buttons now work, CLI unchanged  
**Time:** 30 minutes implementation + 15 minutes documentation  

**Key Insight:** Business logic was already perfect (proved by working CLI). We just needed the bridge.

---

## Related Work

- **TEAM-333:** SSH list command (existing pattern we followed)
- **TEAM-334:** Deleted commands (cleanup that exposed the gap)
- **TEAM-297:** Specta v2 bindings (TypeScript generation system we used)
- **TEAM-293:** Original Tauri wrappers (pattern we restored)

---

**END OF IMPLEMENTATION DOCUMENT**
