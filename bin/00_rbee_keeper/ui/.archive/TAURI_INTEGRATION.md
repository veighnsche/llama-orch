# Tauri Integration - rbee-keeper UI

**Status:** ✅ Connected to Tauri commands with Zustand state management

## Overview

The rbee-keeper UI is now fully connected to the Rust backend via Tauri commands. The CommandsSidebar component triggers Tauri commands that execute the same handler functions used by the CLI. Output is managed through Zustand for reactive state updates and line-by-line display.

## Architecture

```
CommandsSidebar.tsx
    ↓ (onClick)
KeeperPage.tsx
    ↓ (invoke)
Tauri IPC
    ↓
tauri_commands.rs
    ↓
handlers/*.rs (shared with CLI)
    ↓ (NARRATE output)
CommandResponse JSON
    ↓
Zustand Store (commandStore.ts)
    ↓ (reactive updates)
Terminal Display (line-by-line)
```

## State Management

Uses **Zustand** for centralized state management:
- `activeCommand` - Currently executing command
- `isExecuting` - Loading state
- `outputLines` - Array of output lines (streaming-style display)
- Actions: `appendOutput()`, `clearOutput()`, `setActiveCommand()`, etc.

## Available Commands

### Queen Operations
- `queen-start` → `queen_start()` - Start queen-rbee daemon
- `queen-stop` → `queen_stop()` - Stop queen-rbee daemon
- `queen-status` → `queen_status()` - Check queen status
- `queen-info` → `queen_info()` - Get queen info
- `queen-rebuild` → `queen_rebuild()` - Rebuild queen with/without local hive

### Hive Operations (localhost)
- `hive-start` → `hive_start()` - Start rbee-hive on localhost
- `hive-stop` → `hive_stop()` - Stop rbee-hive on localhost
- `hive-status` → `hive_status()` - Check hive status
- `hive-list` → `hive_list()` - List all configured hives

## Response Format

All Tauri commands return a JSON string with this structure:

```typescript
interface CommandResponse {
  success: boolean;
  message: string;
  data?: string;
}
```

## Files Modified

### Frontend (TEAM-294)
- `ui/src/pages/KeeperPage.tsx` - Zustand integration, line-by-line output display
- `ui/src/components/CommandsSidebar.tsx` - Updated command list to match available commands
- `ui/src/store/commandStore.ts` - **NEW** Zustand store for command state
- `ui/package.json` - Added `zustand` dependency

### Backend (TEAM-293)
- `src/tauri_commands.rs` - Tauri command wrappers (already existed)
- `src/main.rs` - Command registration (already existed)

## Usage Example

### Using Zustand Store

```typescript
import { useCommandStore } from "../store/commandStore";

function MyComponent() {
  const { outputLines, appendOutput, clearOutput } = useCommandStore();
  
  // Clear previous output
  clearOutput();
  
  // Append lines (like CLI narration)
  appendOutput("[12:34:56] 🚀 Starting queen...");
  appendOutput("[12:34:57] ✅ Queen started on http://localhost:7833");
  
  // Display lines
  return (
    <div>
      {outputLines.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );
}
```

### Direct Tauri Invoke

```typescript
import { invoke } from "@tauri-apps/api/core";

// Execute a command
const result = await invoke<string>("queen_start");
const response: CommandResponse = JSON.parse(result);

if (response.success) {
  console.log("✅", response.message);
  if (response.data) {
    // Split data into lines for display
    response.data.split("\n").forEach(line => appendOutput(line));
  }
} else {
  console.error("❌", response.message);
}
```

## Testing

1. Start the dev server: `turbo dev --concurrency 16`
2. Open rbee-keeper UI: http://localhost:5173/
3. Click commands in the sidebar
4. Watch output in the terminal window

## Output Display

The UI displays output **line-by-line** similar to CLI narration:
- Each line is a separate `<div>` element
- Empty lines preserved with non-breaking space
- Monospace font for terminal feel
- Copy functionality copies all lines joined with `\n`

**Current Behavior:**
- Commands return a single JSON response
- Response message and data are split into lines
- Lines are appended to `outputLines` array
- UI reactively updates via Zustand

**Future Enhancement:**
For true streaming (like CLI), we need to:
1. Capture stdout from handlers in Rust
2. Emit Tauri events for each narration line
3. Listen to events in frontend
4. Append lines in real-time as they're emitted

## Future Enhancements

- ⭐ **Real-time streaming** - Emit Tauri events for each NARRATE line
- Add command history (previous executions)
- Add command parameters UI (currently hardcoded to localhost)
- Add worker and model management commands
- Add inference UI
- Add auto-scroll to bottom on new output
