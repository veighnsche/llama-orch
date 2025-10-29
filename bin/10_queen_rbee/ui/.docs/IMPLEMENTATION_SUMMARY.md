# iframe Narration Event Bus - Implementation Summary

**Date:** Oct 29, 2025  
**Status:** ✅ IMPLEMENTED - Ready for Testing

## What Was Implemented

### 1. Queen UI (Sender) - `@rbee/queen-rbee-react`

**File:** `packages/queen-rbee-react/src/utils/narrationBridge.ts`
- ✅ `sendNarrationToParent()` - Sends events via postMessage
- ✅ `parseNarrationLine()` - Parses SSE narration format
- ✅ `createNarrationStreamHandler()` - Wraps SSE callback with postMessage

**File:** `packages/queen-rbee-react/src/hooks/useRhaiScripts.ts`
- ✅ Updated `testScript()` to use `QueenClient.submitAndStream()`
- ✅ Integrated `createNarrationStreamHandler()` to send events to parent
- ✅ Operation built as JSON: `{ RhaiScriptTest: { content } }`

**File:** `packages/queen-rbee-react/src/index.ts`
- ✅ Exported narration bridge utilities

### 2. rbee-keeper (Receiver)

**File:** `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts`
- ✅ `setupNarrationListener()` - Listens for postMessage events
- ✅ Origin validation (`http://localhost:7833`)
- ✅ Adds events to `narrationStore`

**File:** `bin/00_rbee_keeper/ui/src/App.tsx`
- ✅ Setup listener on app mount
- ✅ Cleanup on unmount

### 3. Documentation

**File:** `bin/10_queen_rbee/ui/.docs/IFRAME_NARRATION_EVENT_BUS.md`
- ✅ Full architecture documentation
- ✅ Security considerations
- ✅ Message format specification
- ✅ Testing instructions

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│ rbee-keeper (localhost:7834)                            │
│                                                         │
│  App.tsx                                                │
│    ↓ useEffect                                          │
│  setupNarrationListener()                               │
│    ↓ window.addEventListener('message')                 │
│  narrationStore.addEntry(event.payload)                 │
│                          ▲                              │
│                          │ postMessage                  │
│                          │                              │
│  ┌────────────────────────────────────────────────┐   │
│  │ <iframe src="http://localhost:7833" />         │   │
│  │                                                 │   │
│  │  Queen UI                                       │   │
│  │    ↓                                            │   │
│  │  RhaiIDE → Test button                         │   │
│  │    ↓                                            │   │
│  │  useRhaiScripts.testScript(content)            │   │
│  │    ↓                                            │   │
│  │  QueenClient.submitAndStream(operation, cb)    │   │
│  │    ↓                                            │   │
│  │  SSE stream → narrationHandler(line)           │   │
│  │    ↓                                            │   │
│  │  window.parent.postMessage({                   │   │
│  │    type: 'QUEEN_NARRATION',                    │   │
│  │    payload: narrationEvent                     │   │
│  │  }, 'http://localhost:7834')                   │   │
│  └────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Testing Instructions

### 1. Start Services

```bash
# Terminal 1: Start queen-rbee
cd bin/10_queen_rbee
cargo run

# Terminal 2: Start rbee-keeper
cd bin/00_rbee_keeper
cargo run
```

### 2. Open rbee-keeper UI

1. Navigate to `http://localhost:7834`
2. Click "Queen" in sidebar (loads iframe at `http://localhost:7833`)
3. Open browser DevTools console

### 3. Test RHAI Narration

1. In Queen iframe, go to RHAI IDE page
2. Enter test script content (anything)
3. Click "Test" button
4. **Expected Results:**
   - Console shows: `[RHAI Test] SSE line: ...`
   - Console shows: `[RHAI Test] Narration event: {...}`
   - Console shows: `[Keeper] Received narration from Queen: {...}`
   - Narration panel in rbee-keeper shows Queen events

### 4. Verify in Console

**Queen iframe console:**
```
[RHAI Test] SSE line: data: {"actor":"queen-rbee","action":"rhai_test_start",...}
[RHAI Test] Narration event: {actor: "queen-rbee", action: "rhai_test_start", ...}
```

**rbee-keeper console:**
```
[Keeper] Received narration from Queen: {actor: "queen-rbee", action: "rhai_test_start", ...}
```

## Message Format

```typescript
{
  type: 'QUEEN_NARRATION',
  payload: {
    actor: 'queen-rbee',
    action: 'rhai_test_start',
    human: '🧪 Testing RHAI script',
    level: 'Info',
    // ... other fields
  },
  source: 'queen-rbee',
  timestamp: 1730217840000
}
```

## Security

- ✅ Origin validation: Only accepts from `http://localhost:7833`
- ✅ Message type filtering: Only processes `QUEEN_NARRATION`
- ✅ Payload validation: Checks for required fields
- ✅ Safe for localhost development

## Known Limitations

1. **Hardcoded origins** - localhost:7833 and localhost:7834
2. **No encryption** - postMessage is plaintext (OK for localhost)
3. **One-way** - Only Queen → Keeper (can extend if needed)

## Next Steps

1. [ ] Test with actual RHAI operations
2. [ ] Verify narration appears in keeper's narration panel
3. [ ] Test with multiple concurrent operations
4. [ ] Add error handling for malformed messages
5. [ ] Consider bidirectional communication if needed

## Files Modified

### Queen UI
- `packages/queen-rbee-react/src/utils/narrationBridge.ts` (NEW)
- `packages/queen-rbee-react/src/hooks/useRhaiScripts.ts` (MODIFIED)
- `packages/queen-rbee-react/src/index.ts` (MODIFIED)

### rbee-keeper
- `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts` (NEW)
- `bin/00_rbee_keeper/ui/src/App.tsx` (MODIFIED)

### Documentation
- `bin/10_queen_rbee/ui/.docs/IFRAME_NARRATION_EVENT_BUS.md` (NEW)
- `bin/10_queen_rbee/ui/.docs/IMPLEMENTATION_SUMMARY.md` (THIS FILE)

## Troubleshooting

**No events in keeper:**
- Check browser console for CORS errors
- Verify Queen iframe loaded (`http://localhost:7833`)
- Check origin validation in `narrationListener.ts`

**Events not parsed:**
- Check SSE format in `parseNarrationLine()`
- Verify JSON structure matches `NarrationEvent`

**TypeScript errors:**
- Run `pnpm install` in both packages
- Check exports in `index.ts` files
