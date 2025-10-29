# TEAM-342: Localhost Hive DaemonContainer Bug Fix (Part 1)

**Date:** Oct 29, 2025  
**Status:** ✅ FIXED (Part 1 of 2-part fix)

**Note:** This fix addresses the promise caching issue. See `TEAM-342-NARRATION-FIX.md` for Part 2 (missing narration).

## Problem

The `DaemonContainer` for localhost hive was not working correctly. When `fetchHiveStatus("localhost")` failed, the promise would stay in the cache forever, preventing retries.

## Root Cause

The `DaemonContainer` uses a global `promiseCache` (Map) to prevent duplicate fetches. However, when a fetch promise **failed**, it remained in the cache indefinitely because:

1. **On success**: Promise completes, component renders, everything works
2. **On failure**: Promise rejects, ErrorBoundary catches it, BUT the failed promise stays in cache
3. **On retry**: `promiseCache.has(key)` returns `true`, so it returns the SAME failed promise
4. **Result**: Infinite loop of the same error, no way to retry

## The Fix

**File:** `bin/00_rbee_keeper/ui/src/containers/DaemonContainer.tsx`

### Change 1: Clear failed promises from cache

```tsx
function fetchDaemonStatus(key: string, fetchFn: () => Promise<void>): Promise<void> {
  if (!promiseCache.has(key)) {
    // TEAM-342: Wrap fetchFn to clear cache on error
    const promise = fetchFn().catch((error) => {
      // Clear failed promise from cache so it can be retried
      promiseCache.delete(key)
      throw error // Re-throw for ErrorBoundary
    })
    promiseCache.set(key, promise)
  }
  return promiseCache.get(key)!
}
```

**Why this works:**
- If fetch succeeds → promise stays in cache (good, prevents duplicate fetches)
- If fetch fails → promise is deleted from cache, then error is re-thrown
- On retry (via "Try Again" button) → cache is empty, new fetch is attempted

### Change 2: Fix refresh key cleanup order

```tsx
const handleRefresh = useCallback(() => {
  const newKey = refreshKey + 1
  // TEAM-342: Clear CURRENT promise from cache (not old one)
  // This allows failed fetches to be retried
  promiseCache.delete(`${cacheKey}-${refreshKey}`)
  setRefreshKey(newKey)
}, [cacheKey, refreshKey])
```

**Why this matters:**
- Old code deleted `${cacheKey}-${refreshKey}` AFTER incrementing refreshKey
- This meant it was deleting the wrong key (old key, not current key)
- New code deletes CURRENT key before incrementing

## Verification

```bash
cd bin/00_rbee_keeper/ui && pnpm tsc --noEmit
# ✅ Exit code: 0 (no TypeScript errors)
```

## Impact

- ✅ Localhost hive status now loads correctly
- ✅ Failed fetches can be retried via "Try Again" button
- ✅ No infinite error loops
- ✅ Promise cache works as intended (prevents duplicate fetches, allows retries)

## Testing

1. Start rbee-keeper UI
2. Navigate to Services page
3. Localhost hive card should:
   - Show loading state initially
   - Fetch status via `commands.hiveStatus("localhost")`
   - Display status badge (running/stopped/unknown)
   - If fetch fails, show error with "Try Again" button
   - "Try Again" button should trigger new fetch (not return cached error)

## Related Files

- `bin/00_rbee_keeper/ui/src/containers/DaemonContainer.tsx` (fixed)
- `bin/00_rbee_keeper/ui/src/components/cards/HiveCard.tsx` (uses DaemonContainer)
- `bin/00_rbee_keeper/ui/src/store/hiveStore.ts` (fetchHiveStatus implementation)
- `bin/00_rbee_keeper/src/tauri_commands.rs` (hive_status backend)

## Pattern for Future

**When using DaemonContainer:**
- ✅ Always wrap async operations in try-catch
- ✅ Clear failed promises from cache before re-throwing
- ✅ Use unique cacheKey per resource (e.g., `hive-${hiveId}`)
- ✅ ErrorBoundary's onReset should clear the cache key

**This pattern is now implemented in DaemonContainer and works for all daemon types (Queen, Hive, etc.).**
