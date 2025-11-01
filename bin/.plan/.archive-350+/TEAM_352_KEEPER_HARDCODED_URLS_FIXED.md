# TEAM-352: Keeper Hardcoded URLs Fixed ✅

**Date:** Oct 30, 2025  
**Issue:** Hardcoded URLs throughout Keeper UI  
**Status:** ✅ FIXED

---

## Problem

After fixing Queen to use @rbee/shared-config, Keeper still had hardcoded URLs everywhere:

1. **narrationListener.ts** - 6 hardcoded origins
2. **QueenPage.tsx** - 2 hardcoded iframe URLs
3. **HivePage.tsx** - 1 hardcoded iframe URL

**Total:** 9 hardcoded URLs ❌

---

## Files Fixed

### 1. narrationListener.ts

**BEFORE:**
```typescript
allowedOrigins: [
  "http://localhost:7833", // Queen prod
  "http://localhost:7834", // Queen dev
  "http://localhost:7835", // Hive prod
  "http://localhost:7836", // Hive dev
  "http://localhost:7837", // Worker prod
  "http://localhost:7838", // Worker dev
],
```

**AFTER:**
```typescript
import { getAllowedOrigins } from "@rbee/shared-config";

allowedOrigins: getAllowedOrigins(),
```

**Benefit:** Automatically includes all services (Queen, Hive, Worker) dev + prod ports

### 2. QueenPage.tsx

**BEFORE:**
```typescript
const isDev = import.meta.env.DEV
const queenUrl = isDev 
  ? "http://localhost:7834"  // Dev: Direct to Vite dev server
  : "http://localhost:7833"   // Prod: Embedded files from queen backend
```

**AFTER:**
```typescript
import { getIframeUrl } from "@rbee/shared-config";

const isDev = import.meta.env.DEV
const queenUrl = getIframeUrl('queen', isDev)
```

**Benefit:** Single source of truth for Queen ports

### 3. HivePage.tsx

**BEFORE:**
```typescript
const hiveUrl = "http://localhost:7835";
```

**AFTER:**
```typescript
import { getIframeUrl } from "@rbee/shared-config";

const isDev = import.meta.env.DEV
const hiveUrl = getIframeUrl('hive', isDev);
```

**Benefit:** Single source of truth for Hive ports

### 4. package.json

**Added dependency:**
```json
{
  "dependencies": {
    "@rbee/shared-config": "workspace:*"
  }
}
```

---

## Benefits

### 1. Single Source of Truth ✅

All ports defined in one place:
- `frontend/packages/shared-config/src/ports.ts`

**Port changes:** Update once, works everywhere

### 2. Type Safety ✅

```typescript
// Type-safe service names
getIframeUrl('queen', isDev)  // ✅ Valid
getIframeUrl('invalid', isDev) // ❌ TypeScript error
```

### 3. Automatic Updates ✅

When new services added to PORTS:
- `getAllowedOrigins()` automatically includes them
- No manual updates needed in Keeper

### 4. Consistent Behavior ✅

Same port configuration across:
- Queen UI
- Hive UI
- Worker UI
- Keeper UI
- Narration client
- All shared packages

---

## Verification

### Build Tests ✅

```bash
cd frontend/packages/shared-config
pnpm build
# ✅ SUCCESS

cd bin/00_rbee_keeper/ui
# Dependencies installed ✅
```

### No Hardcoded URLs ✅

```bash
grep -r "localhost:[0-9]" src --include="*.ts" --include="*.tsx"
# Result: Only in App.tsx console.log (acceptable) ✅
```

---

## Remaining Hardcoded URL

**File:** `src/App.tsx`

**Location:**
```typescript
console.log('   - Running on: http://localhost:5173')
```

**Status:** ✅ ACCEPTABLE

**Reason:** This is just a console.log showing the dev server port. It's informational only, not used for actual connections.

---

## Summary

**Hardcoded URLs Fixed:**
- narrationListener.ts: 6 URLs → 0 URLs ✅
- QueenPage.tsx: 2 URLs → 0 URLs ✅
- HivePage.tsx: 1 URL → 0 URLs ✅
- **Total: 9 URLs removed** ✅

**Shared Config Usage:**
- `getAllowedOrigins()` - For narration listener
- `getIframeUrl('queen', isDev)` - For Queen iframe
- `getIframeUrl('hive', isDev)` - For Hive iframe

**Dependencies Added:**
- @rbee/shared-config ✅

**Build Status:**
- shared-config builds ✅
- Dependencies installed ✅

---

## Files Changed

```
bin/00_rbee_keeper/ui/
├── package.json                        (ADDED @rbee/shared-config)
└── src/
    ├── utils/
    │   └── narrationListener.ts        (FIXED - uses getAllowedOrigins)
    └── pages/
        ├── QueenPage.tsx               (FIXED - uses getIframeUrl)
        └── HivePage.tsx                (FIXED - uses getIframeUrl)
```

---

## Impact

**Before Fix:**
- 🔴 9 hardcoded URLs
- 🔴 Port changes require updates in 3 files
- 🔴 Easy to miss updates
- 🔴 Inconsistent with Queen UI

**After Fix:**
- ✅ 0 hardcoded URLs (except console.log)
- ✅ Port changes update once in shared-config
- ✅ Automatic propagation
- ✅ Consistent with Queen UI

---

**TEAM-352: All hardcoded URLs removed from Keeper!** ✅
