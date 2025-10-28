# TEAM-336: Narration Panel Not Working - Complete Debugging Guide

**Status:** 🔴 Events not appearing in NarrationPanel

## What Should Happen

1. User opens rbee-keeper GUI
2. Narration panel appears on right side (320px wide)
3. User clicks "Test" button
4. 4 events appear in the panel immediately
5. All `n!()` and `tracing::` events from Rust backend stream to panel

## What's Actually Happening

Events are not appearing in the NarrationPanel.

---

## Step-by-Step Debugging

### STEP 1: Verify Files Exist

```bash
# Check all files were created
ls -la bin/00_rbee_keeper/src/tracing_init.rs
ls -la bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx
ls -la bin/00_rbee_keeper/ui/src/components/Shell.tsx

# Expected output: All 3 files should exist
```

### STEP 2: Rebuild Everything

```bash
cd /home/vince/Projects/llama-orch

# Clean build
cargo clean
cargo build --bin rbee-keeper

# Check for errors
echo "Exit code: $?"
```

**Expected:** Exit code 0 (success)

### STEP 3: Check TypeScript Bindings

```bash
# Regenerate TypeScript bindings
cargo test --package rbee-keeper --lib export_typescript_bindings

# Verify NarrationEvent type exists
grep -A 3 "NarrationEvent" bin/00_rbee_keeper/ui/src/generated/bindings.ts
```

**Expected output:**
```typescript
export type NarrationEvent = { 
  level: string; 
  message: string; 
  timestamp: string 
}
```

### STEP 4: Verify Shell.tsx Imports NarrationPanel

```bash
grep -n "NarrationPanel" bin/00_rbee_keeper/ui/src/components/Shell.tsx
```

**Expected output:**
```
9:import { NarrationPanel } from "./NarrationPanel";
30:        <NarrationPanel />
```

### STEP 5: Check Frontend Build

```bash
cd bin/00_rbee_keeper/ui

# Install dependencies (if needed)
pnpm install

# Build frontend
pnpm build

# Check for errors
echo "Exit code: $?"
```

**Expected:** Exit code 0, no TypeScript errors

### STEP 6: Run with Debug Output

```bash
cd /home/vince/Projects/llama-orch

# Run with full stderr output
RUST_LOG=debug cargo run --bin rbee-keeper 2>&1 | tee rbee-keeper.log
```

**What to look for:**
- GUI window opens
- No panic messages
- Narration panel visible on right side

### STEP 7: Open Browser DevTools

1. Launch rbee-keeper GUI
2. Press **F12** to open DevTools
3. Go to **Console** tab
4. Look for these messages:

**Expected console output:**
```
[NarrationPanel] Setting up listener for 'narration' events
```

**If you DON'T see this:** React component not mounting properly

### STEP 8: Click Test Button

1. Find "Narration" panel on right side
2. Click **"Test"** button in header
3. Watch console for:

**Expected console output:**
```
[NarrationPanel] Test result: "Narration test events emitted - check the panel!"
[NarrationPanel] Received event: { level: "INFO", message: "🎯 Test narration event from Tauri command", timestamp: "2025-10-28T13:43:00.123Z" }
[NarrationPanel] Received event: { level: "INFO", message: "This is a tracing::info! event", timestamp: "..." }
[NarrationPanel] Received event: { level: "WARN", message: "This is a tracing::warn! event", timestamp: "..." }
[NarrationPanel] Received event: { level: "ERROR", message: "This is a tracing::error! event", timestamp: "..." }
```

### STEP 9: Check Rust Stderr

Look at the `rbee-keeper.log` file or stderr output:

**Expected stderr output:**
```
INFO rbee_keeper: 🎯 Test narration event from Tauri command
INFO rbee_keeper: This is a tracing::info! event
WARN rbee_keeper: This is a tracing::warn! event
ERROR rbee_keeper: This is a tracing::error! event
```

**If you see this in stderr but NOT in console:** Tauri event emission failing

---

## Common Issues & Fixes

### Issue 1: NarrationPanel Not Visible

**Symptoms:** No panel on right side of window

**Debug:**
```bash
# Check Shell.tsx has NarrationPanel
grep "NarrationPanel" bin/00_rbee_keeper/ui/src/components/Shell.tsx
```

**Fix:** Verify Shell.tsx looks like this:
```tsx
import { NarrationPanel } from "./NarrationPanel";

export function Shell({ children }: ShellProps) {
  return (
    <div className="fixed inset-0 flex flex-col bg-background text-foreground">
      <CustomTitlebar />
      <div className="flex-1 flex overflow-hidden">
        <KeeperSidebar />
        <main className="flex-1 overflow-y-auto">{children}</main>
        <NarrationPanel />  {/* ← Must be here */}
      </div>
    </div>
  );
}
```

### Issue 2: Console Shows "Setting up listener" But No Events

**Symptoms:** Listener registered, but clicking Test does nothing

**Debug:**
```javascript
// In browser console, manually test Tauri invoke:
const { invoke } = await import('@tauri-apps/api/core');
const result = await invoke('test_narration');
console.log(result);
```

**Expected:** Should return success message

**If it fails:** Tauri command not registered or crashing

### Issue 3: Events in Stderr But Not in Browser Console

**Symptoms:** Rust logs show events, but frontend doesn't receive them

**Root Cause:** Tauri Emitter not working

**Debug in Rust:**
```rust
// Add to tracing_init.rs on_event() method:
fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
    let mut visitor = EventVisitor::default();
    event.record(&mut visitor);

    let payload = NarrationEvent {
        level: event.metadata().level().to_string(),
        message: visitor.message.clone(),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // DEBUG: Print to stderr before emitting
    eprintln!("🔍 About to emit: {:?}", payload);
    
    match self.app_handle.emit("narration", &payload) {
        Ok(_) => eprintln!("✅ Emit succeeded"),
        Err(e) => eprintln!("❌ Emit failed: {:?}", e),
    }
}
```

### Issue 4: Tracing Subscriber Not Initialized

**Symptoms:** No events at all, not even in stderr

**Debug:**
```bash
# Check main.rs has init_gui_tracing() in setup
grep -A 5 "\.setup" bin/00_rbee_keeper/src/main.rs
```

**Expected:**
```rust
.setup(|app| {
    rbee_keeper::init_gui_tracing(app.handle().clone());
    Ok(())
})
```

### Issue 5: Frontend Build Out of Sync

**Symptoms:** Old code running, changes not visible

**Fix:**
```bash
cd bin/00_rbee_keeper/ui
rm -rf dist/
pnpm build

cd /home/vince/Projects/llama-orch
cargo build --bin rbee-keeper
```

### Issue 6: TypeScript Import Errors

**Symptoms:** Console shows module not found errors

**Debug:**
```bash
# Check bindings file exists and has NarrationEvent
cat bin/00_rbee_keeper/ui/src/generated/bindings.ts | grep -A 3 NarrationEvent
```

**Fix:** Regenerate bindings:
```bash
cargo test --package rbee-keeper --lib export_typescript_bindings
```

---

## Manual Event Test (Bypass Everything)

If nothing works, test Tauri events directly:

### 1. Add Simple Test Command

```rust
// In tauri_commands.rs
#[tauri::command]
#[specta::specta]
pub async fn simple_test(app: tauri::AppHandle) -> Result<String, String> {
    use tauri::Emitter;
    
    eprintln!("🔍 simple_test called");
    
    let payload = serde_json::json!({
        "level": "INFO",
        "message": "Direct test event",
        "timestamp": "2025-10-28T13:43:00Z"
    });
    
    match app.emit("narration", &payload) {
        Ok(_) => {
            eprintln!("✅ Direct emit succeeded");
            Ok("Event emitted".to_string())
        }
        Err(e) => {
            eprintln!("❌ Direct emit failed: {:?}", e);
            Err(format!("Failed: {:?}", e))
        }
    }
}
```

### 2. Call from Browser Console

```javascript
const { invoke } = await import('@tauri-apps/api/core');
await invoke('simple_test');
```

**If this works:** Problem is in tracing layer
**If this fails:** Problem is in Tauri event system

---

## Nuclear Option: Start Fresh

If nothing works, verify the entire setup:

### 1. Check File Contents

```bash
# Shell.tsx should have NarrationPanel
cat bin/00_rbee_keeper/ui/src/components/Shell.tsx

# NarrationPanel.tsx should exist
cat bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx | head -50

# tracing_init.rs should have TauriNarrationLayer
cat bin/00_rbee_keeper/src/tracing_init.rs | grep -A 20 "impl.*Layer"

# main.rs should call init_gui_tracing
cat bin/00_rbee_keeper/src/main.rs | grep -A 5 "init_gui_tracing"
```

### 2. Verify All Imports

```bash
# Check NarrationPanel imports
grep "^import" bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx

# Check Shell imports
grep "^import" bin/00_rbee_keeper/ui/src/components/Shell.tsx
```

### 3. Check Cargo.toml Dependencies

```bash
grep -A 2 "chrono" bin/00_rbee_keeper/Cargo.toml
grep -A 2 "tracing-subscriber" bin/00_rbee_keeper/Cargo.toml
```

**Expected:**
```toml
tracing-subscriber = { version = "0.3", features = ["env-filter", "fmt", "registry"] }
chrono = { version = "0.4", features = ["serde"] }
```

---

## Expected File Structure

```
bin/00_rbee_keeper/
├── src/
│   ├── main.rs                    # ✅ Has init_gui_tracing() in .setup()
│   ├── lib.rs                     # ✅ Exports tracing_init module
│   ├── tracing_init.rs            # ✅ NEW - Custom tracing layer
│   └── tauri_commands.rs          # ✅ Has test_narration command
├── ui/
│   └── src/
│       ├── components/
│       │   ├── Shell.tsx          # ✅ Imports and renders NarrationPanel
│       │   └── NarrationPanel.tsx # ✅ NEW - Right panel component
│       └── generated/
│           └── bindings.ts        # ✅ Has NarrationEvent type
└── Cargo.toml                     # ✅ Has chrono + tracing-subscriber deps
```

---

## Quick Checklist

Run through this checklist:

- [ ] `cargo build --bin rbee-keeper` succeeds
- [ ] `cargo test --package rbee-keeper --lib export_typescript_bindings` succeeds
- [ ] `NarrationEvent` type exists in `ui/src/generated/bindings.ts`
- [ ] `Shell.tsx` imports and renders `<NarrationPanel />`
- [ ] `NarrationPanel.tsx` exists and has `listen("narration", ...)` 
- [ ] `tracing_init.rs` has `TauriNarrationLayer` with `emit()` call
- [ ] `main.rs` calls `init_gui_tracing()` in `.setup()` hook
- [ ] `test_narration` command registered in `invoke_handler`
- [ ] Frontend builds without errors (`cd ui && pnpm build`)
- [ ] GUI launches without crashes
- [ ] Narration panel visible on right side (320px wide)
- [ ] Browser console shows listener setup message
- [ ] Clicking "Test" button triggers console logs

**If ALL boxes checked but still not working:** There's a deeper Tauri/React issue

---

## Last Resort: Add Verbose Logging

### In tracing_init.rs:

```rust
fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
    eprintln!("🔍 [TauriNarrationLayer] on_event called");
    
    let mut visitor = EventVisitor::default();
    event.record(&mut visitor);
    
    eprintln!("🔍 [TauriNarrationLayer] Visitor message: {:?}", visitor.message);

    let payload = NarrationEvent {
        level: event.metadata().level().to_string(),
        message: visitor.message,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    eprintln!("🔍 [TauriNarrationLayer] Payload: {:?}", payload);
    
    match self.app_handle.emit("narration", &payload) {
        Ok(_) => eprintln!("✅ [TauriNarrationLayer] Emit succeeded"),
        Err(e) => eprintln!("❌ [TauriNarrationLayer] Emit failed: {:?}", e),
    }
}
```

### In NarrationPanel.tsx:

```typescript
useEffect(() => {
    console.log("🔍 [NarrationPanel] useEffect running");
    console.log("🔍 [NarrationPanel] About to call listen()");
    
    const unlisten = listen<NarrationEvent>("narration", (event) => {
        console.log("🔍 [NarrationPanel] Event received!");
        console.log("🔍 [NarrationPanel] Event payload:", event.payload);
        console.log("🔍 [NarrationPanel] Current entries count:", entries.length);
        
        setEntries((prev) => {
            const newEntries = [...prev, { ...event.payload, id: idCounter.current++ }];
            console.log("🔍 [NarrationPanel] New entries count:", newEntries.length);
            return newEntries;
        });
    });

    console.log("🔍 [NarrationPanel] Listener registered");
    
    return () => {
        console.log("🔍 [NarrationPanel] Cleanup running");
        unlisten.then((fn) => fn());
    };
}, []);
```

Run again and watch **both** stderr and browser console for the 🔍 emoji logs.

---

## Contact Points

If still not working after all this:

1. **Stderr logs:** Should show 🔍 emoji logs from Rust
2. **Browser console:** Should show 🔍 emoji logs from TypeScript
3. **Gap between them:** Indicates where the pipeline breaks

**Most likely culprits:**
1. Tauri event system not initialized properly
2. React component not mounting
3. TypeScript bindings out of sync
4. Frontend build not running

---

**TEAM-336** | **Debug Guide** | **Date:** 2025-10-28
