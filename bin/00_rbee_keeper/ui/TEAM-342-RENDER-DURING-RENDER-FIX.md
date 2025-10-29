# TEAM-342: React Render-During-Render Fix

**Date:** Oct 29, 2025  
**Status:** ✅ FIXED

## Problem

React error in console:
```
Cannot update a component (`KeeperSidebar`) while rendering a different component (`DaemonFetcher`). 
To locate the bad setState() call inside `DaemonFetcher`, follow the stack trace...
```

## Root Cause

The `fetchHives` and `fetchHiveStatus` functions in the Zustand store were calling `set()` **synchronously** to cache promises:

```tsx
// OLD CODE (BROKEN)
const promise = (async () => { ... })();

set((state) => {
  state._fetchHivesPromise = promise;  // ❌ Triggers re-render during render
});
return promise;
```

**The flow:**
1. `DaemonContainer` renders
2. Calls `use(fetchDaemonStatus(...))` which calls `fetchHiveStatus()`
3. `fetchHiveStatus()` immediately calls `set()` to cache the promise
4. `set()` triggers a Zustand store update
5. Store update triggers re-render of `KeeperSidebar` (which uses the store)
6. **React error:** Can't update `KeeperSidebar` while rendering `DaemonFetcher`

## The Fix

**File:** `bin/00_rbee_keeper/ui/src/store/hiveStore.ts`

Use `queueMicrotask()` to defer the `set()` call until after the current render completes:

```tsx
// NEW CODE (FIXED)
const promise = (async () => { ... })();

// TEAM-342: Defer promise caching to avoid render-during-render
queueMicrotask(() => {
  set((state) => {
    state._fetchHivesPromise = promise;  // ✅ Deferred until after render
  });
});
return promise;
```

**Why this works:**
- `queueMicrotask()` schedules the callback to run **after** the current JavaScript execution completes
- This means the `set()` call happens **after** React finishes rendering
- No more "setState during render" error

## Changes Made

1. ✅ `fetchHives()` - Deferred promise caching with `queueMicrotask()`
2. ✅ `fetchHiveStatus()` - Deferred promise caching with `queueMicrotask()`

## Why Not Use Direct Mutation?

**Attempted fix (WRONG):**
```tsx
get()._fetchPromises.set(hiveId, promise);  // ❌ Breaks reactivity
```

This bypasses Immer's proxy, so React won't detect the change. The store won't re-render when needed.

**Correct fix:**
```tsx
queueMicrotask(() => {
  set((state) => {
    state._fetchPromises.set(hiveId, promise);  // ✅ Goes through Immer
  });
});
```

This uses Immer's proxy, so React properly tracks the change.

## Verification

```bash
cd bin/00_rbee_keeper/ui && pnpm tsc --noEmit
# ✅ Exit code: 0 (no TypeScript errors)
```

## Testing

1. Start rbee-keeper UI
2. Navigate to Services page
3. **Before fix:** Console error about render-during-render
4. **After fix:** No console errors, smooth rendering

## Pattern for Future

**When caching promises in Zustand stores:**
- ❌ **NEVER** call `set()` synchronously in a function that might be called during render
- ✅ **ALWAYS** defer `set()` calls with `queueMicrotask()` when caching promises
- ✅ Return the promise immediately (don't await the microtask)

**Example pattern:**
```tsx
const fetchSomething = async () => {
  const existing = get()._promise;
  if (existing) return existing;

  const promise = (async () => {
    // ... async work ...
  })();

  // Defer caching to avoid render-during-render
  queueMicrotask(() => {
    set((state) => {
      state._promise = promise;
    });
  });

  return promise;
};
```

## Related Issues

This is **Part 3** of the localhost hive fix:
- **Part 1:** Promise caching bug in `DaemonContainer` (TEAM-342-LOCALHOST-HIVE-FIX.md)
- **Part 2:** Missing narration in `hive_status` command (TEAM-342-NARRATION-FIX.md)
- **Part 3:** Render-during-render error (this document)

All three issues are now fixed.
