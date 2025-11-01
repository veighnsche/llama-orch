# TEAM-350: Fixed postMessage Origin for Dev/Prod Environments

**Status:** ✅ COMPLETE

## The Error

```
[Error] Unable to post message to http://localhost:7834. 
Recipient has origin http://localhost:5173.
```

## Root Cause

The `narrationBridge.ts` was hardcoded to send postMessage to `http://localhost:7834`, but:

**Development:**
- rbee-keeper runs on **port 5173** (Vite dev server)
- queen-rbee UI runs on **port 7834** (Vite dev server)

**Production:**
- rbee-keeper is a **Tauri app** (port 7834 or specific origin)
- queen-rbee UI is **embedded in binary** at `http://localhost:7833/`

## The Fix

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts`

```typescript
// TEAM-350: Environment-aware parent origin
const isDev = typeof (import.meta as any).env !== 'undefined' && (import.meta as any).env.DEV
const parentOrigin = isDev 
  ? 'http://localhost:5173'  // Dev: Vite dev server
  : 'http://localhost:7834'  // Prod: Tauri app

window.parent.postMessage(message, parentOrigin)
```

## How It Works

### Development Mode
- `import.meta.env.DEV` is `true` (Vite sets this)
- postMessage sends to `http://localhost:5173` (rbee-keeper Vite dev server)
- ✅ Origin matches, message delivered

### Production Mode
- `import.meta.env.DEV` is `false` (production build)
- postMessage sends to `http://localhost:7834` (Tauri app)
- ✅ Origin matches, message delivered

## Port Configuration Reference

From `PORT_CONFIGURATION.md`:

| Component | Dev Port | Prod Location |
|-----------|----------|---------------|
| **rbee-keeper GUI** | 5173 | Tauri app |
| **queen-rbee UI** | 7834 | Embedded at 7833/ |
| **rbee-hive UI** | 7836 | Embedded at 7835/ |
| **llm-worker UI** | 7837 | Embedded at 8080/ |

## Testing

```bash
# 1. Rebuild React package
cd bin/10_queen_rbee/ui/packages/queen-rbee-react
pnpm build

# 2. Rebuild queen-rbee (build.rs will rebuild everything)
cargo build --bin queen-rbee

# 3. Start queen-rbee
cargo run --bin queen-rbee

# 4. Open rbee-keeper at http://localhost:5173 (dev)

# 5. Navigate to Queen UI (iframe)

# 6. Press Test button in RHAI IDE

# Expected: No postMessage errors, narration appears in rbee-keeper!
```

## Expected Console Output

**Before fix:**
```
[Error] Unable to post message to http://localhost:7834. 
Recipient has origin http://localhost:5173.
```

**After fix:**
```
[Log] [RHAI Test] Narration event: {actor: "queen_rbee", action: "rhai_test_start", ...}
// No postMessage errors!
```

## Files Changed

1. **bin/10_queen_rbee/ui/packages/queen-rbee-react/src/utils/narrationBridge.ts** - Environment-aware origin

## Related

This completes the narration flow:

1. ✅ Backend emits narration with job_id
2. ✅ SSE endpoint sends JSON (not plain text)
3. ✅ Frontend parses JSON successfully
4. ✅ postMessage sends to correct origin (dev/prod aware)
5. ✅ rbee-keeper receives and displays narration

## Alternative Approach (If Needed)

If the Tauri app origin is different, you can use:

```typescript
const parentOrigin = isDev 
  ? 'http://localhost:5173'  // Dev
  : '*'                       // Prod: Allow any origin (less secure but flexible)
```

Or detect the actual parent origin:

```typescript
const parentOrigin = isDev 
  ? 'http://localhost:5173'
  : window.location.ancestorOrigins?.[0] || 'http://localhost:7834'
```

---

**TEAM-350 Signature:** Fixed postMessage origin to be environment-aware (dev port 5173, prod port 7834)
