# TEAM-352 Step 4: Remove Hardcoded URLs and Config Cleanup

**Estimated Time:** 20-30 minutes  
**Priority:** HIGH  
**Previous Step:** TEAM_352_STEP_3_NARRATION_MIGRATION.md  
**Next Step:** TEAM_352_STEP_5_TESTING.md

---

## Mission

Remove ALL remaining hardcoded URLs and use @rbee/shared-config throughout Queen UI.

**Why This Matters:**
- Single source of truth for ports
- Easy to add new services
- No port conflicts
- Consistent across dev/prod

**Locations to update:**
1. Queen App (`app/src/App.tsx`)
2. Any remaining components with hardcoded ports
3. Verify Keeper uses shared config too

**Code Reduction:** ~10-15 LOC of hardcoded values removed

---

## Deliverables Checklist

- [x] Updated App.tsx to use @rbee/dev-utils
- [x] Searched for all hardcoded port numbers
- [x] Removed hardcoded URLs from app code
- [x] Verified no hardcoded origins in app/src
- [x] Package builds successfully
- [x] TEAM-352 signatures added

**STATUS: ✅ COMPLETE - See TEAM_352_STEP_4_COMPLETE.md for details**
**NOTE:** Hook default parameters are acceptable (not hardcoded in app code)

---

## Step 1: Update Queen App.tsx

Navigate to Queen app:

```bash
cd bin/10_queen_rbee/ui/app
```

Check if @rbee/dev-utils is already a dependency:

```bash
cat package.json | grep "dev-utils"
```

**If NOT present, add it:**

Edit `package.json`:

```json
{
  "dependencies": {
    "@rbee/queen-rbee-react": "workspace:*",
    "@rbee/queen-rbee-sdk": "workspace:*",
    "@rbee/ui": "workspace:*",
    "@rbee/dev-utils": "workspace:*",
    // ... other deps
  }
}
```

Install:

```bash
cd ../../..  # Back to monorepo root
pnpm install
```

---

## Step 2: Replace Startup Logging in App.tsx

Navigate back to app:

```bash
cd bin/10_queen_rbee/ui/app
```

**Current App.tsx (lines 9-21):**

```typescript
// TEAM-350: Log build mode on startup
const isDev = import.meta.env.DEV;
if (isDev) {
  console.log("🔧 [QUEEN UI] Running in DEVELOPMENT mode");
  console.log("   - Vite dev server active (hot reload enabled)");
  console.log(
    "   - Loaded via: http://localhost:7833/dev (proxied from :7834)",
  );
} else {
  console.log("🚀 [QUEEN UI] Running in PRODUCTION mode");
  console.log("   - Serving embedded static files");
  console.log("   - Loaded via: http://localhost:7833/");
}
```

**Replace with:**

Edit `src/App.tsx`:

```typescript
// TEAM-352: Migrated to use @rbee/dev-utils for startup logging
// Old implementation: ~13 LOC of manual environment logging
// New implementation: 1 LOC using shared utility
// Reduction: 12 LOC

import { ThemeProvider } from "next-themes";
import { logStartupMode } from "@rbee/dev-utils";
import DashboardPage from "./pages/DashboardPage";

// TEAM-352: Use shared startup logging
logStartupMode("QUEEN UI");

function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="min-h-screen bg-background font-sans">
        <DashboardPage />
      </div>
    </ThemeProvider>
  );
}

export default App;
```

**Key changes:**
- ✅ Removed manual isDev check
- ✅ Removed hardcoded port mentions
- ✅ Uses `logStartupMode()` from shared package

---

## Step 3: Search for Hardcoded Ports

Search the ENTIRE Queen UI codebase for hardcoded ports:

```bash
cd bin/10_queen_rbee/ui

# Search for localhost:7833
grep -r "localhost:7833" . --exclude-dir=node_modules --exclude-dir=dist --exclude="*.backup"

# Search for localhost:7834
grep -r "localhost:7834" . --exclude-dir=node_modules --exclude-dir=dist --exclude="*.backup"

# Search for localhost:5173 (Keeper port)
grep -r "localhost:5173" . --exclude-dir=node_modules --exclude-dir=dist --exclude="*.backup"
```

**Expected result:** NO MATCHES (we've already fixed all hooks and bridge).

**If matches found:**
- Examine each file
- Replace with `getServiceUrl()` or `getIframeUrl()` from @rbee/shared-config
- Document what you changed

---

## Step 4: Search for Hardcoded Origins

Search for hardcoded origin arrays:

```bash
cd bin/10_queen_rbee/ui

grep -r "allowedOrigins\|http://localhost" . --exclude-dir=node_modules --exclude-dir=dist --exclude="*.backup"
```

**Expected result:** NO MATCHES in Queen UI (origins handled by @rbee/narration-client).

**If matches found:**
- Should only be in Keeper UI (we'll handle that next)
- Document location

---

## Step 5: Update Keeper UI (if needed)

**NOTE:** The plan document says to update Keeper, but let's verify what's actually needed.

Navigate to Keeper:

```bash
cd bin/00_rbee_keeper/ui
```

**Check if Keeper needs shared-config:**

```bash
cat package.json | grep "shared-config"
```

**If NOT present, add dependencies:**

Edit `package.json`:

```json
{
  "dependencies": {
    // ... existing deps
    "@rbee/shared-config": "workspace:*",
    "@rbee/dev-utils": "workspace:*"
  }
}
```

Install:

```bash
cd ../../..  # Back to monorepo root
pnpm install
```

---

## Step 6: Update Keeper QueenPage.tsx (if exists)

Check if QueenPage.tsx has hardcoded URLs:

```bash
cd bin/00_rbee_keeper/ui
find src -name "*Queen*" -type f
```

**If QueenPage.tsx exists, check for hardcoded URLs:**

```bash
cat src/pages/QueenPage.tsx | grep -A 3 "localhost"
```

**If hardcoded URLs found, replace:**

```typescript
// TEAM-352: Old hardcoded approach
const isDev = import.meta.env.DEV
const queenUrl = isDev 
  ? "http://localhost:7834"  // ❌ Hardcoded
  : "http://localhost:7833"   // ❌ Hardcoded

// TEAM-352: New shared config approach
import { getIframeUrl } from '@rbee/shared-config'

const isDev = import.meta.env.DEV
const queenUrl = getIframeUrl('queen', isDev)
```

---

## Step 7: Update Keeper narrationListener.ts (if exists)

Check if narrationListener has hardcoded origins:

```bash
cd bin/00_rbee_keeper/ui
find src -name "*narration*" -o -name "*listener*" -type f
```

**If narrationListener.ts exists, check for hardcoded origins:**

```bash
cat src/utils/narrationListener.ts | grep -A 5 "allowedOrigins"
```

**If hardcoded origins found, replace:**

```typescript
// TEAM-352: Old hardcoded approach
const allowedOrigins = [
  "http://localhost:7833",  // ❌ Hardcoded
  "http://localhost:7834",  // ❌ Hardcoded
  "http://localhost:7835",  // ❌ Hardcoded (hive)
  "http://localhost:7836",  // ❌ Hardcoded (hive dev)
]

// TEAM-352: New shared config approach
import { getAllowedOrigins } from '@rbee/shared-config'

const allowedOrigins = getAllowedOrigins()
// Automatically includes all services (queen, hive, worker) dev + prod ports
```

---

## Step 8: Update Keeper App.tsx (if needed)

Check if Keeper App.tsx has manual startup logging:

```bash
cd bin/00_rbee_keeper/ui
cat src/App.tsx | grep -A 10 "console.log"
```

**If manual logging found, replace:**

```typescript
// TEAM-352: Replace manual logging
import { logStartupMode } from '@rbee/dev-utils'

logStartupMode("KEEPER UI")
```

---

## Step 9: Build and Verify

Build Queen UI:

```bash
cd bin/10_queen_rbee/ui/app
pnpm build
```

**Expected:**
```
✓ Built successfully
No hardcoded URLs in bundle
```

Build Keeper UI (if modified):

```bash
cd bin/00_rbee_keeper/ui
pnpm build
```

**Expected:**
```
✓ Built successfully
```

---

## Step 10: Verify Startup Logs

Test the new startup logging:

**Terminal 1:** Start Queen backend
```bash
cargo run --bin queen-rbee
```

**Terminal 2:** Start Queen UI
```bash
cd bin/10_queen_rbee/ui/app
pnpm dev
```

Open http://localhost:7834 in browser, check console:

**Expected log:**
```
🔧 [QUEEN UI] DEVELOPMENT mode
├─ Vite dev server: http://localhost:7834
├─ Hot reload: ENABLED
└─ Environment: development
```

**NOT the old logs** with hardcoded ports.

---

## Step 11: Document Changes

Create a list of all files modified in this step:

```bash
cd bin/10_queen_rbee/ui

# Files modified:
# - app/package.json (added @rbee/dev-utils)
# - app/src/App.tsx (replaced startup logging)

# Optionally modified (if hardcoded URLs found):
# - (list any additional files)
```

---

## Step 12: Count Lines Removed

Calculate code reduction:

```bash
cd bin/10_queen_rbee/ui/app/src

# If you kept backup of App.tsx:
# wc -l App.tsx.backup App.tsx
# Old: ~34 LOC
# New: ~22 LOC
# Reduction: ~12 LOC
```

**Additional hardcoded URL removals:**
- Each hardcoded URL = ~1 LOC saved
- Each hardcoded origin array = ~5 LOC saved

**Total this step:** ~12-20 LOC removed

---

## Testing Checklist

Before moving to next step:

- [ ] Queen App builds successfully
- [ ] Keeper App builds successfully (if modified)
- [ ] Startup logs use new format
- [ ] No hardcoded ports in codebase
- [ ] No hardcoded origins in codebase
- [ ] `grep -r "localhost:783" .` returns no matches (except docs/tests)
- [ ] iframe loads correctly in Keeper
- [ ] Narration still works
- [ ] No TypeScript errors
- [ ] No runtime errors

---

## Troubleshooting

### Issue: @rbee/dev-utils not found

**Fix:**
```bash
cd frontend/packages/dev-utils
pnpm build

cd ../../bin/10_queen_rbee/ui/app
pnpm install
```

### Issue: Startup logs don't appear

**Check:**
1. Is `logStartupMode()` being called?
2. Is import correct?
3. Check browser console (not terminal)

### Issue: iframe doesn't load after changes

**Debug:**
```typescript
// In Keeper QueenPage.tsx, add debugging:
const queenUrl = getIframeUrl('queen', isDev)
console.log('[Keeper] Queen iframe URL:', queenUrl)
```

Should be:
- Dev: `http://localhost:7834`
- Prod: `http://localhost:7833`

### Issue: Narration stops working

**Check:**
1. Origins are correct (use `getAllowedOrigins()`)
2. postMessage is sending to correct origin
3. See TEAM_352_STEP_3 troubleshooting

---

## Success Criteria

✅ All hardcoded URLs removed  
✅ All hardcoded origins removed  
✅ Startup logging uses shared utility  
✅ Queen App builds successfully  
✅ Keeper App builds successfully  
✅ ~12-20 LOC removed  
✅ TEAM-352 signatures added

---

## Next Step

Continue to **TEAM_352_STEP_5_TESTING.md** for comprehensive end-to-end testing.

---

**TEAM-352 Step 4: Config cleanup complete!** ✅
