# Daemon Container Refactor

**TEAM-339:** Extracted generic daemon data provider pattern from QueenContainer

## Problem

`QueenContainer.tsx` implemented a React 19 Suspense-based data loading pattern that was specific to Queen but could be reused for Hive cards and other daemon services.

## Solution (Rule Zero Applied)

Created a white-labeled `DaemonContainer` component that can be reused across all daemon services (Queen, Hive, etc.).

**Rule Zero Decision:** Deleted the wrapper files (`QueenContainer.tsx`, `HiveContainer.tsx`) instead of keeping them for backwards compatibility. Callers now use `DaemonContainer` directly with inline props.

## Files

### Created: `src/containers/DaemonContainer.tsx` (167 LOC)
- Generic daemon data provider using React 19 `use()` hook
- Handles loading/error states with Suspense
- Promise caching for performance
- Error boundary with retry functionality
- Customizable metadata (name, description)
- Pluggable fetch function

### Deleted: `src/containers/QueenContainer.tsx` (Rule Zero)
- **Before:** 158 LOC with all logic inline
- **Refactored to:** 34 LOC thin wrapper
- **Final decision:** DELETED - wrapper added no value (entropy)
- Callers now use DaemonContainer directly

### Deleted: `src/containers/HiveContainer.tsx` (Rule Zero)
- **Created as:** 40 LOC thin wrapper
- **Final decision:** DELETED - wrapper added no value (entropy)
- Callers now use DaemonContainer directly

## Usage Pattern (After Rule Zero)

### Queen
```tsx
<DaemonContainer
  cacheKey="queen"
  metadata={{
    name: "Queen",
    description: "Smart API server",
  }}
  fetchFn={() => useQueenStore.getState().fetchStatus()}
>
  <QueenCard />
</DaemonContainer>
```

### Hive
```tsx
<DaemonContainer
  cacheKey={`hive-${hiveId}`}
  metadata={{
    name: title,
    description: description,
  }}
  fetchFn={() => useSshHivesStore.getState().fetchHiveStatus(hiveId)}
>
  <HiveCard hiveId={hiveId} />
</DaemonContainer>
```

### Any Daemon
```tsx
<DaemonContainer
  cacheKey="my-daemon"
  metadata={{ name: "My Daemon", description: "..." }}
  fetchFn={() => myStore.getState().fetchStatus()}
>
  <MyDaemonCard />
</DaemonContainer>
```

## Benefits

1. **Code Reuse:** 158 LOC → 0 LOC for Queen (100% elimination via direct use)
2. **No Entropy:** Deleted wrappers instead of keeping for "compatibility"
3. **Consistency:** All daemon cards use same loading/error pattern
4. **Maintainability:** Bug fixes in one place benefit all daemons
5. **Extensibility:** Easy to add new daemon types (workers, models, etc.)
6. **Type Safety:** Generic props with TypeScript support
7. **Rule Zero:** Breaking changes > backwards compatibility (pre-1.0)

## Architecture (After Rule Zero)

```
DaemonContainer (generic - 167 LOC)
├── Promise caching
├── Error boundary
├── Suspense wrapper
└── Loading fallback

Callers (inline props)
├── Queen: DaemonContainer + inline metadata
├── Hive: DaemonContainer + inline metadata
└── Others: DaemonContainer + inline metadata
```

**No wrappers = No entropy**

## Key Features

- **React 19 Suspense:** No useEffect, pure declarative data fetching
- **Promise Caching:** Prevents duplicate fetches on re-renders
- **Error Recovery:** Try Again button clears cache and retries
- **Loading States:** Customizable loading fallbacks
- **Type Safety:** Full TypeScript support with generics

## Migration Path

1. ✅ Extract generic DaemonContainer (167 LOC)
2. ✅ Refactor QueenContainer to use DaemonContainer (158 → 34 LOC)
3. ✅ Create HiveContainer using DaemonContainer (40 LOC)
4. ✅ **Rule Zero Applied:** Delete wrappers, use DaemonContainer directly
5. ✅ Update ServicesPage to use DaemonContainer with inline props
6. 🔲 Update HiveCard usage as needed (optional)

## Rule Zero Decision

**Why we deleted the wrappers:**
- 34 LOC wrapper (QueenContainer) = entropy
- 40 LOC wrapper (HiveContainer) = entropy
- Pre-1.0 = license to break things
- Inline props = clearer, no indirection
- Compiler will find all call sites (30 seconds to fix)
- Entropy is permanent, breaking changes are temporary

**Result:** 158 LOC → 0 LOC (100% elimination)
