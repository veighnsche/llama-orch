# TEAM-336: Tracing Consolidation & GUI Narration Streaming

**Status:** ✅ COMPLETE

## Mission

Eliminate repeated tracing initialization code and implement proper narration streaming for the Tauri GUI.

## Problem

Duplicate tracing setup in both `handle_command()` and `launch_gui()` functions in main.rs:

```rust
// REPEATED CODE (12 lines × 2 = 24 lines)
fmt()
    .with_writer(std::io::stderr)
    .with_ansi(true)
    .with_line_number(false)
    .with_file(false)
    .with_target(false)
    .with_env_filter(EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info")))
    .init();
```

**Different requirements:**
- **CLI:** Narration to stderr only (for terminal users)
- **GUI:** Narration to stderr (debugging) + Tauri events (React sidebar)

## Solution

Created `src/tracing_init.rs` with two functions:

### 1. CLI Tracing (stderr only)

```rust
pub fn init_cli_tracing() {
    fmt()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_line_number(false)
        .with_file(false)
        .with_target(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}
```

### 2. GUI Tracing (stderr + Tauri events)

```rust
pub fn init_gui_tracing(app_handle: tauri::AppHandle) {
    // Layer 1: stderr output for debugging
    let stderr_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_line_number(false)
        .with_file(false)
        .with_target(false);

    // Layer 2: Tauri event emitter for React sidebar
    let tauri_layer = TauriNarrationLayer::new(app_handle);

    // Combine layers
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(stderr_layer)
        .with(tauri_layer)
        .init();
}
```

### Custom Tracing Layer

Implements `tracing_subscriber::Layer` to capture narration events and emit them to Tauri frontend:

```rust
impl<S> Layer<S> for TauriNarrationLayer
where
    S: tracing::Subscriber,
    S: for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        // Extract event message
        let payload = NarrationEvent {
            level: event.metadata().level().to_string(),
            message: visitor.message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        // Emit to Tauri frontend (non-blocking)
        let _ = self.app_handle.emit("narration", &payload);
    }
}
```

## Frontend Integration

TypeScript bindings automatically generated in `ui/src/generated/bindings.ts`:

```typescript
/**
 * Narration event payload for Tauri frontend
 */
export type NarrationEvent = { 
  level: string; 
  message: string; 
  timestamp: string 
}
```

React component can listen to narration events:

```typescript
import { listen } from '@tauri-apps/api/event';
import type { NarrationEvent } from './generated/bindings';

// Listen to all narration events from backend
const unlisten = await listen<NarrationEvent>('narration', (event) => {
  console.log(`[${event.payload.level}] ${event.payload.message}`);
  // Add to sidebar, display in UI, format with timestamp, etc.
});

// Clean up listener when component unmounts
return () => unlisten();
```

## Files Changed

### NEW
- `bin/00_rbee_keeper/src/tracing_init.rs` (113 LOC)
  - `init_cli_tracing()` - Simple stderr output
  - `init_gui_tracing()` - Dual output (stderr + Tauri events)
  - `TauriNarrationLayer` - Custom tracing layer
  - `NarrationEvent` - Event payload with specta support

### MODIFIED
- `bin/00_rbee_keeper/src/main.rs`
  - `handle_command()`: 12 lines → 1 line (`init_cli_tracing()`)
  - `launch_gui()`: 12 lines → 1 line in `.setup()` hook (`init_gui_tracing()`)
- `bin/00_rbee_keeper/src/lib.rs`
  - Added `pub mod tracing_init;`
  - Re-exported functions for convenience
- `bin/00_rbee_keeper/src/tauri_commands.rs`
  - Added `NarrationEvent` to TypeScript bindings export
- `bin/00_rbee_keeper/Cargo.toml`
  - Added `chrono` dependency for timestamps
  - Added `registry` feature to `tracing-subscriber`

## Code Reduction

- **Before:** 24 lines of duplicated setup code
- **After:** 2 lines (function calls)
- **Savings:** 22 lines in main.rs, centralized in reusable module

## Benefits

✅ **No duplication** - Single source of truth for tracing setup  
✅ **Type-safe events** - `NarrationEvent` with specta TypeScript bindings  
✅ **Dual output for GUI** - Debugging (stderr) + User-facing (React sidebar)  
✅ **Real-time narration** - All `n!()` macro calls stream to frontend  
✅ **Non-blocking** - Event emission never blocks main thread  
✅ **Automatic timestamps** - ISO 8601 format via `chrono`

## Architecture

### CLI Flow
```
n!() macro → tracing event → stderr → Terminal user sees it
```

### GUI Flow
```
n!() macro → tracing event → TauriNarrationLayer
                           ├→ stderr (for debugging)
                           └→ Tauri emit("narration") → React sidebar
```

## Tauri v2 API

**Key insight:** In Tauri v2, `AppHandle` implements the `Emitter` trait. Must import:

```rust
use tauri::Emitter;

// Then can use:
app_handle.emit("event-name", &payload)?;
```

**NOT `emit_all()` like in v1!** Just `emit()` for global events.

## Next Steps

Frontend team can now:
1. Import generated `NarrationEvent` type
2. Listen to `narration` events
3. Display in real-time sidebar
4. Filter by level (info/warn/error)
5. Add timestamps, icons, formatting

## Compilation

```bash
cargo check --bin rbee-keeper
# ✅ PASS (69 warnings from narration-core deprecations, not our code)
```

## Testing

### CLI Mode
```bash
# Should show narration on stderr
cargo run --bin rbee-keeper -- status
```

### GUI Mode
```bash
# Build and run
cargo build --bin rbee-keeper
cargo run --bin rbee-keeper

# In the GUI:
# 1. Look for the "Narration" panel on the right side
# 2. Click the "Test" button in the panel header
# 3. Should see 4 events appear:
#    - 🎯 Test narration event from Tauri command
#    - This is a tracing::info! event
#    - This is a tracing::warn! event
#    - This is a tracing::error! event
```

### Browser Console Debugging
Open DevTools (F12) and check for:
```
[NarrationPanel] Setting up listener for 'narration' events
[NarrationPanel] Received event: { level: "INFO", message: "...", timestamp: "..." }
```

## Troubleshooting

If events don't appear:
1. Check browser console for listener setup message
2. Click "Test" button - should emit 4 events
3. Check stderr for tracing output
4. Verify `init_gui_tracing()` is called in `.setup()` hook

See `TEAM_336_NARRATION_PANEL_DEBUG.md` for detailed debugging guide.

---

**TEAM-336** | **Date:** 2025-10-28 | **Status:** Complete
