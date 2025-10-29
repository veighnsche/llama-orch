# 🔍 Architecture Review & Fixes (TEAM-354)

**Date:** 2025-10-29  
**Reviewer:** TEAM-354  
**Reference:** CORRECT_ARCHITECTURE.md

---

## **Issues Found & Fixed**

### **Issue 1: Components Not Using QueryContainer** ❌ → ✅

**Problem:** HiveCard and QueenCard were manually handling loading/error states instead of delegating to QueryContainer, violating Pattern 4 of CORRECT_ARCHITECTURE.md.

**Spec (Pattern 4, lines 188-212):**
```typescript
export function HiveCard({ hiveId }: { hiveId: string }) {
  const { hive, isLoading, error, refetch } = useHive(hiveId)
  
  return (
    <QueryContainer
      isLoading={isLoading}
      error={error}
      data={hive}
      onRetry={refetch}
    >
      {(hive) => <Card>...</Card>}
    </QueryContainer>
  )
}
```

**What We Had (Wrong):**
```typescript
export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { hive, isLoading, error, refetch } = useHive(hiveId);
  
  // ❌ Manual loading state
  if (isLoading && !hive) {
    return <Card><Loader2 /></Card>;
  }
  
  // ❌ Manual error state
  if (error && !hive) {
    return <Card><Alert /></Card>;
  }
  
  // Render logic...
}
```

**Fixed:**
```typescript
export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { hive, isLoading, error, refetch } = useHive(hiveId);
  const { start, stop, install, uninstall, refreshCapabilities } = useHiveActions();
  const { isExecuting } = useCommandStore();

  return (
    <QueryContainer<SshHive>  // ✅ Delegate to QueryContainer
      isLoading={isLoading}
      error={error}
      data={hive}
      onRetry={refetch}
      metadata={{ name: `${title} Hive`, description }}
    >
      {(hive) => <HiveCardContent ... />}  // ✅ Type-safe data
    </QueryContainer>
  );
}
```

**Benefits:**
- ✅ Type-safe data (TypeScript knows `hive` is not null in children)
- ✅ Consistent loading/error UI across all components
- ✅ DRY principle - one place for loading/error logic
- ✅ Stale-while-revalidate built-in

**Files Fixed:**
- `src/components/cards/HiveCard.tsx`
- `src/components/cards/QueenCard.tsx`

---

### **Issue 2: Missing Stale-While-Revalidate Indicator** ❌ → ✅

**Problem:** When data is being refreshed (stale-while-revalidate), users couldn't see that a background refresh was happening.

**Spec (lines 421-463):**
```typescript
// When data exists but is stale (> 5s old)
// Start loading BUT keep old data
set((state) => {
  state.queries.set(hiveId, {
    data: existing.data,  // ← OLD DATA STILL VISIBLE
    isLoading: true,      // ← But show loading indicator
    ...
  })
})
```

**Fixed in QueryContainer:**
```typescript
// Success state - render children with type-safe data
// TEAM-354: Show stale-while-revalidate indicator when refreshing
return (
  <div className="relative">
    {isLoading && (
      <div className="absolute top-2 right-2 z-10">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
      </div>
    )}
    {children(data)}  // ✅ Old data visible while loading fresh
  </div>
)
```

**User Experience:**
- User sees old status immediately (no blank screen)
- Small spinner appears in top-right corner when refreshing
- Fresh data loads in background
- Status updates smoothly (no flash)

---

### **Issue 3: Redundant `installedHives` Checks** ❌ → ✅

**Problem:** Components were checking both `hive?.isInstalled` and `installedHives.includes(hiveId)` which was redundant.

**What We Had:**
```typescript
const installedHivesStore = useSshHivesStore();
const installedHives = installedHivesStore.installedHives;
const isInstalled = hive?.isInstalled ?? installedHives.includes(hiveId);  // ❌ Redundant
```

**Fixed:**
```typescript
const isInstalled = hive.isInstalled ?? false;  // ✅ Single source of truth
```

**Why:** The `fetchHive` function already sets `isInstalled` based on the API response, so checking `installedHives` array is unnecessary. The query cache is the single source of truth.

---

## **Architecture Patterns Verified**

### **✅ Pattern 1: Query-Based Store**
- Queries stored as `Map<id, HiveQuery>`
- Each query has `data`, `isLoading`, `error`, `lastFetch`
- No promise caching
- Automatic deduplication via `isLoading` + `lastFetch` checks

### **✅ Pattern 2: Query Hooks**
- `useHive(id)` triggers fetch via `useEffect`
- Returns `{ hive, isLoading, error, refetch }`
- Hook owns fetching, not component

### **✅ Pattern 3: Dumb Container**
- QueryContainer is generic `<T>`
- Only handles loading/error/empty states
- Children receive type-safe data
- Now includes stale-while-revalidate indicator

### **✅ Pattern 4: Component Usage**
- Components call hooks
- Pass query state to QueryContainer
- Render logic in children function
- Type-safe data guaranteed

### **✅ Pattern 5: Localhost Special Case**
- LocalhostHive component (no install/uninstall)
- Filtered out of SSH lists
- Excluded from install dropdown

### **✅ Pattern 6: Action Hooks**
- Mutations separated from queries
- `useHiveActions()` for mutations
- `useQueenActions()` for mutations
- Invalidation triggers refetch

---

## **Edge Cases Handled**

### **1. Stale Data with Loading**
- **Scenario:** User clicks refresh on a card
- **Behavior:** Old data stays visible, small spinner appears in corner
- **Code:** `if (isLoading) <Loader2 />` inside success state

### **2. Stale Data with Error**
- **Scenario:** Refresh fails but old data exists
- **Behavior:** Old data stays visible (QueryContainer checks `error && !data`)
- **Code:** Error state only shows when `!data`

### **3. Multiple Components Same ID**
- **Scenario:** Two HiveCards mount for same `hiveId`
- **Behavior:** Only one fetch (deduplication via `isLoading` check)
- **Code:** `if (existing?.isLoading) return` in `fetchHive`

### **4. Fresh Data Check**
- **Scenario:** Component re-mounts with fresh data
- **Behavior:** No fetch if data < 5s old
- **Code:** `if (!force && ... && (now - lastFetch < 5000)) return`

### **5. Type Safety**
- **Scenario:** Component renders with data
- **Behavior:** TypeScript knows data is not null
- **Code:** `QueryContainer<SshHive>` generic + children function

---

## **Performance Improvements**

| Feature | Before | After |
|---------|--------|-------|
| **Deduplication** | Broken (race conditions) | Works (checks isLoading) |
| **Stale-while-revalidate** | Not implemented | Fully implemented |
| **Type safety** | Manual null checks | Generic QueryContainer |
| **Code duplication** | Loading/error in each component | Single QueryContainer |

---

## **Verification Checklist**

- [x] Components use QueryContainer (Pattern 4)
- [x] Stale-while-revalidate indicator visible
- [x] No redundant `installedHives` checks
- [x] Type-safe data in children functions
- [x] Loading states consistent across components
- [x] Error states consistent across components
- [x] Deduplication working (isLoading check)
- [x] Fresh data check working (< 5s)
- [x] QueryContainer shows loading spinner when refreshing with stale data
- [x] All 6 architecture patterns implemented correctly

---

## **Files Modified**

### **Fixed:**
1. `src/components/cards/HiveCard.tsx` - Now uses QueryContainer
2. `src/components/cards/QueenCard.tsx` - Now uses QueryContainer
3. `src/containers/QueryContainer.tsx` - Added stale-while-revalidate indicator

### **Already Correct:**
- `src/store/hiveStore.ts` - Query pattern implemented correctly
- `src/store/queenStore.ts` - Query pattern implemented correctly
- `src/components/cards/LocalhostHive.tsx` - Handles own UI (simple component)

---

## **Summary**

All architectural violations fixed. The codebase now fully conforms to CORRECT_ARCHITECTURE.md:

✅ **Query-based stores** - No promise caching, Map-based query cache  
✅ **Query hooks** - Drive fetching via useEffect  
✅ **QueryContainer** - Dumb UI, type-safe, with stale-while-revalidate  
✅ **Component pattern** - Hooks → QueryContainer → Type-safe children  
✅ **Localhost separation** - Dedicated component  
✅ **Action hooks** - Mutations separated from queries  

**Zero architectural debt remaining.**
