# TEAM-350: Fixed SSE to Send JSON Instead of Plain Text

**Status:** ✅ COMPLETE

## The Problem We Found

SSE endpoint was sending **formatted plain text** instead of **JSON**:

**What was being sent (WRONG):**
```
"\u001b[1mqueen_rbee::rhai::test::...\u001b[0m \u001b[2mrhai_test_start\u001b[0m"
"🧪 Testing RHAI script"
"[DONE]"
```

**What should be sent (CORRECT):**
```json
{"actor":"queen_rbee","action":"rhai_test_start","message":"🧪 Testing RHAI script","timestamp":"...","job_id":"...","formatted":"..."}
{"actor":"queen_rbee","action":"rhai_test_content","message":"Script length: 0 bytes","timestamp":"...","job_id":"...","formatted":"..."}
"[DONE]"
```

## Root Cause

**File:** `bin/10_queen_rbee/src/http/jobs.rs` line 145

```rust
// OLD CODE (WRONG)
yield Ok(Event::default().data(&event.formatted));
```

The SSE endpoint was only sending the `formatted` field (plain text for terminal display), not the full JSON object.

## The Fix

```rust
// NEW CODE (CORRECT)
// TEAM-350: Send JSON for frontend parsing (not formatted text)
// Frontend needs structured data to display in UI
let json = serde_json::to_string(&event)
    .unwrap_or_else(|_| event.formatted.clone());
yield Ok(Event::default().data(&json));
```

Now the entire `NarrationEvent` struct is serialized to JSON:

```rust
pub struct NarrationEvent {
    pub formatted: String,      // For terminal display
    pub actor: String,          // For programmatic access
    pub action: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub job_id: Option<String>,
    pub correlation_id: Option<String>,
}
```

## Why This Matters

### Frontend Needs Structured Data

The `narrationBridge.ts` tries to parse each line as JSON:

```typescript
export function createNarrationStreamHandler(
  onEvent?: (event: NarrationEvent) => void
): (line: string) => void {
  return (line: string) => {
    try {
      const event = JSON.parse(line) as NarrationEvent  // ← Needs JSON!
      
      // Send to parent window (rbee-keeper)
      if (window.parent !== window) {
        window.parent.postMessage({
          type: 'NARRATION_EVENT',
          payload: event
        }, 'http://localhost:7834')
      }
      
      onEvent?.(event)
    } catch (e) {
      console.warn('[Queen] Failed to parse narration line:', line, e)
    }
  }
}
```

**Before fix:** All lines failed to parse (plain text)
**After fix:** Lines parse successfully (JSON)

## Expected Behavior After Fix

### Console Logs

```
[RHAI Test] Starting test...
[RHAI Test] Client created, baseUrl: "http://localhost:7833"
[RHAI Test] Operation: {operation: "rhai_script_test", content: ""}
[RHAI Test] Submitting and streaming...

[RHAI Test] SSE line: {"actor":"queen_rbee","action":"rhai_test_start",...}
[RHAI Test] Narration event: {actor: "queen_rbee", action: "rhai_test_start", ...}

[RHAI Test] SSE line: {"actor":"queen_rbee","action":"rhai_test_content",...}
[RHAI Test] Narration event: {actor: "queen_rbee", action: "rhai_test_content", ...}

[RHAI Test] SSE line: {"actor":"queen_rbee","action":"rhai_test_error",...}
[RHAI Test] Narration event: {actor: "queen_rbee", action: "rhai_test_error", ...}

[RHAI Test] SSE line: [DONE]
[RHAI Test] Stream complete, receivedDone: true
✅ Test completed successfully
```

### No More Parse Errors

**Before:**
```
[Warning] [Queen] Failed to parse narration line: "🧪 Testing RHAI script"
SyntaxError: JSON Parse error: Unrecognized token '🧪'
```

**After:**
```
[Log] [RHAI Test] Narration event: {
  actor: "queen_rbee",
  action: "rhai_test_start",
  message: "🧪 Testing RHAI script",
  timestamp: "2025-10-29T19:00:00Z",
  job_id: "abc123",
  formatted: "..."
}
```

### Narration Reaches rbee-keeper

The narration bridge will now successfully:
1. Parse JSON events ✅
2. Send to parent window via postMessage ✅
3. rbee-keeper receives and displays in narration store ✅

## Testing

```bash
# 1. Rebuild queen-rbee
cargo build --bin queen-rbee

# 2. Start queen-rbee
cargo run --bin queen-rbee

# 3. Open http://localhost:7834 (rbee-keeper with queen iframe)

# 4. Navigate to RHAI IDE

# 5. Press Test button

# 6. Check console - should see JSON events, no parse errors

# 7. Check rbee-keeper narration panel - should see events appear!
```

## Files Changed

1. **bin/10_queen_rbee/src/http/jobs.rs** - Send JSON instead of formatted text

## Related Files (No Changes Needed)

- `narrationBridge.ts` - Already expects JSON ✅
- `NarrationEvent` struct - Already has Serialize ✅
- rbee-keeper message handler - Already expects structured data ✅

## Why It Was Plain Text Before

The `formatted` field was added (TEAM-201) for **terminal display**:
- Colored text for stdout
- Human-readable format
- Good for `cargo run` debugging

But the **SSE endpoint** should send **JSON** for:
- Frontend parsing
- Structured data access
- Cross-window communication

The fix maintains both:
- `formatted` field still exists (for terminal)
- Full JSON sent over SSE (for frontend)

---

**TEAM-350 Signature:** Fixed SSE endpoint to send JSON narration events instead of plain text
