# Phase 4: Cleanup (1 day)

**Team:** TEAM-353  
**Duration:** 1 day  
**Status:** 🔴 NOT STARTED  
**Dependencies:** Phase 3 must be complete

---

## **Goal**

Remove all hacks, dead code, and legacy patterns. Final polish.

---

## **Tasks**

### **Task 1: Remove Store Loading/Error State** (2 hours)

**Rationale:** Query cache owns this now, stores don't need it

**File:** `src/store/hiveStore.ts`

```typescript
// BEFORE (Phase 2)
interface HiveStoreState {
  queries: Map<string, HiveQuery>  // ← Has loading/error
  sshHives: SshHive[]
  sshHivesLoading: boolean         // ← REMOVE THIS
  sshHivesError: string | null     // ← REMOVE THIS
}

// AFTER (Phase 4)
interface SshHiveListQuery {
  data: SshHive[]
  isLoading: boolean
  error: string | null
  lastFetch: number
}

interface HiveStoreState {
  queries: Map<string, HiveQuery>      // Individual hive queries
  sshHivesQuery: SshHiveListQuery      // SSH list query
}
```

**File:** `src/store/queenStore.ts`

```typescript
// BEFORE (Phase 2)
interface QueenStoreState {
  query: QueenQuery  // Already correct ✅
}

// AFTER (Phase 4)
// No changes needed - already using query pattern
```

---

### **Task 2: Remove Immer enableMapSet** (30 min)

**File:** `src/store/hiveStore.ts`

```typescript
// TEAM-353: Remove Immer hack (no longer needed)
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
// import { enableMapSet } from 'immer'  // ← DELETE THIS
// enableMapSet()  // ← DELETE THIS

// Map<string, HiveQuery> works fine with Immer by default
export const useHiveStore = create<HiveStoreState>()(
  immer((set, get) => ({
    queries: new Map(),  // ← Works without enableMapSet
    // ...
  }))
)
```

**Why this works now:**
- Old: Stored `Map<string, Promise<void>>` → needed enableMapSet
- New: Store `Map<string, HiveQuery>` → Immer handles it automatically

---

### **Task 3: Remove queueMicrotask Hacks** (1 hour)

**Search for all occurrences:**

```bash
cd bin/00_rbee_keeper/ui
grep -r "queueMicrotask" src/
```

**Expected:** None found (already removed in Phase 2)

**If found:** Delete them (they were band-aids for promise caching)

---

### **Task 4: Remove Dead Code** (2 hours)

**Files to check:**

```bash
# Check for unused imports
pnpm exec eslint src/ --fix

# Check for unused exports
pnpm exec ts-prune
```

**Common dead code:**
- Old container components (should be deleted in Phase 3)
- Unused helper functions
- Commented-out code
- TODO markers from previous teams

**Rule:** If it's not used, delete it (Rule Zero)

---

### **Task 5: Update Documentation** (2 hours)

**Files to update:**

**File:** `README.md`

```markdown
# Bee Keeper UI

Desktop app for managing Queen and Hive services.

## Architecture

- **Stores:** Zustand with query-based pattern (React Query style)
- **Hooks:** `useHive(id)`, `useQueen()`, `useSshHives()`
- **Containers:** Generic `QueryContainer<T>` for loading/error states
- **Components:** Type-safe, composable, simple

## Key Patterns

### Fetching Data

```typescript
// Hook drives fetching automatically
const { hive, isLoading, error, refetch } = useHive('localhost')

// Container handles UI states
<QueryContainer isLoading={isLoading} error={error} data={hive}>
  {(hive) => <YourComponent hive={hive} />}
</QueryContainer>
```

### Localhost vs SSH Hives

- **Localhost:** Always available (no installation)
- **SSH Hives:** Require installation from SSH config
```

**File:** `src/store/README.md` (NEW)

```markdown
# Store Architecture

## Query Pattern

Stores use query-based pattern (inspired by React Query):

```typescript
interface Query<T> {
  data: T | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}

interface StoreState {
  queries: Map<string, Query<T>>
  fetch: (id: string, force?: boolean) => Promise<void>
}
```

## Features

- **Automatic deduplication:** Multiple calls to same fetch = 1 request
- **Stale-while-revalidate:** Show old data while fetching new
- **Type-safe:** Query type matches data type
- **No promises in state:** Only serializable data

## Usage

```typescript
// Store
export const useHiveStore = create<HiveStoreState>()(
  immer((set, get) => ({
    queries: new Map(),
    fetchHive: async (id, force) => { /* ... */ }
  }))
)

// Hook
export function useHive(id: string) {
  const store = useHiveStore()
  useEffect(() => { store.fetchHive(id) }, [id])
  return store.queries.get(id) ?? { /* defaults */ }
}

// Component
const { hive, isLoading, error } = useHive('localhost')
```
```

---

### **Task 6: Final Testing** (3 hours)

**Test Checklist:**

- [ ] **Localhost hive**
  - [ ] Start works without installation
  - [ ] Stop works
  - [ ] Status badge updates
  - [ ] No "Install" button shown
  
- [ ] **SSH hives**
  - [ ] Install from dropdown works
  - [ ] Start/Stop installed hive works
  - [ ] Uninstall works
  - [ ] Localhost not in dropdown
  
- [ ] **Deduplication**
  - [ ] Mount 3 HiveCards → only 1 fetch
  - [ ] Navigate away and back → uses cache
  - [ ] Force refresh works
  
- [ ] **Error handling**
  - [ ] Network error shows error UI
  - [ ] Retry button works
  - [ ] Error doesn't break other queries
  
- [ ] **Loading states**
  - [ ] Loading spinner shows initially
  - [ ] Stale data shown while refetching
  - [ ] Smooth transitions

**Performance:**
```bash
# Check bundle size
pnpm run build
# Should be smaller than before (less code)

# Check for memory leaks
# Use Chrome DevTools Performance tab
# Record: Open app → navigate pages → close app
# Should not leak queries/promises
```

---

## **Checklist**

- [ ] Remove `isLoading`/`error` from stores (use query cache)
- [ ] Remove Immer `enableMapSet()`
- [ ] Verify no `queueMicrotask` in codebase
- [ ] Delete dead code (unused imports, components)
- [ ] Update `README.md` with new architecture
- [ ] Create `src/store/README.md` explaining query pattern
- [ ] Run ESLint and fix all warnings
- [ ] Run TypeScript compiler (strict mode)
- [ ] All tests pass
- [ ] Performance tested (no memory leaks)
- [ ] Code reviewed by team lead

---

## **Success Criteria**

✅ Zero `queueMicrotask` in codebase  
✅ Zero `enableMapSet` in codebase  
✅ Zero duplicate loading/error state  
✅ Zero dead code (ESLint clean)  
✅ Zero TypeScript errors (strict mode)  
✅ All tests pass  
✅ Documentation updated  
✅ Performance metrics good

---

## **Final Metrics**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total LOC | ~1000 | ~600 | **40% reduction** |
| Promise hacks | 4 files | 0 files | **100% removed** |
| Container complexity | 161 LOC | 40 LOC | **75% simpler** |
| Type safety | Partial | Full | **100% type-safe** |
| Deduplication | Broken | Works | **100% reliable** |

---

## **Handoff**

After Phase 4 completion:

1. **Merge to main** (all phases complete)
2. **Tag release** (`v0.2.0-query-refactor`)
3. **Update team docs** (share query pattern guide)
4. **Knowledge transfer** (demo to other teams)
5. **Monitor production** (watch for regressions)

**Future work:**
- Consider adding mutation tracking (optimistic updates)
- Consider adding query invalidation on WebSocket events
- Consider persisting query cache to localStorage
