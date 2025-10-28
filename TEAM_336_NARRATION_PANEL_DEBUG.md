# TEAM-336: Narration Panel Debugging Guide

## Issue
Narration events are not appearing in the NarrationPanel component.

## Debugging Steps

### 1. Check Browser Console
Open the app and check the browser console for:
- `[NarrationPanel] Setting up listener for 'narration' events` - confirms listener is registered
- `[NarrationPanel] Received event: {...}` - confirms events are being received

### 2. Test Event Emission
Trigger some actions that should emit narration:
```bash
# From the GUI, try:
# - Starting queen (queen_start command)
# - Stopping queen (queen_stop command)
# - SSH list (ssh_list command)
```

### 3. Check Rust Backend
The tracing layer should emit events. Check stderr output:
```bash
# Run rbee-keeper and watch stderr
cargo run --bin rbee-keeper 2>&1 | grep -i narration
```

### 4. Verify Tauri Event System
Test if Tauri events work at all:

**Add to any Tauri command:**
```rust
use tauri::Emitter;

app_handle.emit("test-event", "Hello from Rust")?;
```

**Listen in frontend:**
```typescript
listen("test-event", (event) => {
  console.log("Test event received:", event.payload);
});
```

### 5. Check Tracing Subscriber Setup
Verify `init_gui_tracing()` is called in `.setup()` hook:

```rust
// In main.rs launch_gui()
.setup(|app| {
    rbee_keeper::init_gui_tracing(app.handle().clone());
    Ok(())
})
```

## Common Issues

### Issue: No events at all
**Cause:** Tracing subscriber not initialized
**Fix:** Verify `init_gui_tracing()` is called in Tauri `.setup()` hook

### Issue: Events in stderr but not in frontend
**Cause:** Tauri Emitter trait not imported or emit() failing silently
**Fix:** Check that `use tauri::Emitter;` is present in tracing_init.rs

### Issue: Listener not receiving events
**Cause:** Event name mismatch
**Fix:** Verify both sides use `"narration"` (case-sensitive)

### Issue: Empty messages
**Cause:** EventVisitor not capturing fields correctly
**Fix:** Updated to capture all fields, not just "message" field

## Current Implementation

### Rust Side (tracing_init.rs)
```rust
// Custom layer captures tracing events
impl<S> Layer<S> for TauriNarrationLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = EventVisitor::default();
        event.record(&mut visitor);

        let payload = NarrationEvent {
            level: event.metadata().level().to_string(),
            message: visitor.message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Emit to frontend
        let _ = self.app_handle.emit("narration", &payload);
    }
}
```

### Frontend Side (NarrationPanel.tsx)
```typescript
useEffect(() => {
    console.log("[NarrationPanel] Setting up listener");
    
    const unlisten = listen<NarrationEvent>("narration", (event) => {
        console.log("[NarrationPanel] Received:", event.payload);
        setEntries((prev) => [...prev, { ...event.payload, id: idCounter.current++ }]);
    });

    return () => unlisten.then((fn) => fn());
}, []);
```

## Testing Narration Manually

Add a test command to verify the pipeline:

```rust
// In tauri_commands.rs
#[tauri::command]
#[specta::specta]
pub async fn test_narration() -> Result<String, String> {
    use observability_narration_core::n;
    
    n!("test", "🎯 Test narration event from Tauri command");
    
    Ok("Narration test event emitted".to_string())
}
```

Then call from frontend:
```typescript
import { commands } from './generated/bindings';

await commands.testNarration();
// Check if event appears in NarrationPanel
```

## Expected Behavior

When working correctly:
1. User triggers action (e.g., clicks "Start Queen")
2. Tauri command executes
3. Rust code calls `n!()` macro
4. Tracing event created
5. `TauriNarrationLayer` captures event
6. Event emitted via `app_handle.emit("narration", payload)`
7. Frontend listener receives event
8. NarrationPanel updates with new entry
9. Entry appears in right sidebar with timestamp, level, message

## Next Steps

1. Open browser DevTools console
2. Trigger an action (e.g., SSH list)
3. Check for console logs
4. If no logs appear, tracing subscriber might not be initialized
5. If logs appear but no events, check Tauri event system
6. If events received but empty messages, check EventVisitor implementation
