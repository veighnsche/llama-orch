# 🗑️ REMOVAL PLAN - What to Delete

**RULE ZERO:** Delete complexity, don't add compatibility layers

---

## **Phase 1: Delete Localhost from SSH Logic**

### **Files to Modify**

**File:** `src/components/InstalledHiveList.tsx`

**REMOVE:**
```typescript
// Lines 19-22: DELETE THIS LOGIC
const hasLocalhost = installedHives.includes("localhost");
const localhostInConfig = hives.some((h) => h.host === "localhost");
const showLocalhost = hasLocalhost && !localhostInConfig;

// Lines 32-38: DELETE THIS COMPONENT
{showLocalhost && (
  <HiveCard
    hiveId="localhost"
    title="localhost"
    description="This machine"
  />
)}
```

**REPLACE WITH:**
```typescript
// Filter out localhost entirely
const installedSshHives = hives.filter(
  (hive) => installedHives.includes(hive.host) && hive.host !== 'localhost'
)
```

---

**File:** `src/components/cards/InstallHiveCard.tsx`

**REMOVE:**
```typescript
// Lines 63-66: DELETE LOCALHOST FROM DROPDOWN
{!isLocalhostInstalled && (
  <SelectItem value="localhost">
    <SshTargetItem name="localhost" subtitle="This machine" />
  </SelectItem>
)}
```

**REPLACE WITH:**
```typescript
// Filter out localhost from available targets
availableHives.filter((hive) => hive.host !== 'localhost')
```

---

## **Phase 2: Delete Promise Caching**

### **Files to Rewrite (Delete and Replace)**

**File:** `src/store/hiveStore.ts`

**DELETE:**
```typescript
// Line 8: DELETE THIS IMPORT
import { enableMapSet } from "immer";

// Line 14: DELETE THIS CALL
enableMapSet();

// Lines 31-32: DELETE PROMISE CACHE
_fetchHivesPromise: Promise<void> | null;
_fetchPromises: Map<string, Promise<void>>;

// Lines 29-30: DELETE DUPLICATE STATE
isLoading: boolean;
error: string | null;

// Lines 102-106: DELETE queueMicrotask HACK
queueMicrotask(() => {
  set((state) => {
    state._fetchHivesPromise = promise;
  });
});

// Lines 150-155: DELETE queueMicrotask HACK
queueMicrotask(() => {
  set((state) => {
    state._fetchPromises.set(hiveId, promise);
  });
});
```

**REPLACE WITH:**
```typescript
// Query cache pattern
queries: Map<string, HiveQuery>

interface HiveQuery {
  data: SshHive | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}
```

---

**File:** `src/store/queenStore.ts`

**DELETE:**
```typescript
// Lines 17-19: DELETE DUPLICATE STATE
isLoading: boolean;
error: string | null;

// Line 20: DELETE PROMISE CACHE
_fetchPromise: Promise<void> | null;
```

**REPLACE WITH:**
```typescript
// Query pattern
query: QueenQuery

interface QueenQuery {
  data: QueenStatus | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}
```

---

## **Phase 3: Delete DaemonContainer**

### **Files to Delete Completely**

```bash
# DELETE THESE FILES (Rule Zero)
rm src/containers/DaemonContainer.tsx        # 161 LOC deleted
rm src/containers/SshHivesContainer.tsx      # If exists
```

### **Files to Modify (Remove Imports)**

**Files:** All component files using DaemonContainer

**REMOVE:**
```typescript
import { DaemonContainer } from '@/containers/DaemonContainer'
```

**REPLACE WITH:**
```typescript
import { QueryContainer } from '@/containers/QueryContainer'
```

---

**File:** `src/components/cards/HiveCard.tsx`

**DELETE:**
```typescript
// Lines 167-184: DELETE WRAPPER COMPONENT
export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  return (
    <DaemonContainer
      cacheKey={`hive-${hiveId}`}
      metadata={{ name: title, description: description }}
      fetchFn={() => useSshHivesStore.getState().fetchHiveStatus(hiveId)}
    >
      <HiveCardContent hiveId={hiveId} title={title} description={description} />
    </DaemonContainer>
  )
}

// Lines 30-165: DELETE INNER COMPONENT
function HiveCardContent({ hiveId, title, description }: HiveCardProps) {
  const { hives, installedHives, isLoading, start, stop, /* ... */ } = useSshHivesStore()
  // ... 135 lines of component logic
}
```

**REPLACE WITH:**
```typescript
// Single component using QueryContainer
export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { hive, isLoading, error, refetch, start, stop } = useHive(hiveId)
  
  return (
    <QueryContainer isLoading={isLoading} error={error} data={hive}>
      {(hive) => (
        <Card>{/* render logic */}</Card>
      )}
    </QueryContainer>
  )
}
```

---

## **Phase 4: Delete Hacks and Dead Code**

### **Search and Destroy**

```bash
# Find all queueMicrotask (should be ZERO after Phase 2)
grep -r "queueMicrotask" src/

# Find all enableMapSet (should be ZERO after Phase 2)
grep -r "enableMapSet" src/

# Find all _fetchPromise (should be ZERO after Phase 2)
grep -r "_fetchPromise" src/

# Find all TODO markers from previous teams
grep -r "// TODO" src/

# Find commented-out code
grep -r "^[[:space:]]*//.*" src/ | grep -v "TEAM-"
```

### **Files to Clean**

**File:** `src/store/commandUtils.ts`

**REVIEW:** May not be needed anymore (check if used)

**If unused:**
```bash
rm src/store/commandUtils.ts
```

---

## **Deletion Summary**

| Phase | Files Deleted | Lines Deleted | Pattern Removed |
|-------|--------------|---------------|-----------------|
| 1 | 0 | ~30 LOC | Localhost in SSH lists |
| 2 | 0 | ~150 LOC | Promise caching |
| 3 | 2 files | ~350 LOC | DaemonContainer pattern |
| 4 | Varies | ~70 LOC | Hacks, dead code |
| **Total** | **2-3 files** | **~600 LOC** | **40% reduction** |

---

## **Verification After Deletion**

```bash
# 1. TypeScript must compile cleanly
pnpm run type-check

# 2. ESLint must pass
pnpm run lint

# 3. Tests must pass
pnpm run test

# 4. Build must succeed
pnpm run build

# 5. App must run
pnpm run dev
```

**If any fail:** You deleted something you shouldn't have. Revert and investigate.

---

## **Safety Checklist**

Before deleting any file:

- [ ] Check for imports (use IDE "Find References")
- [ ] Check for tests referencing it
- [ ] Check for documentation referencing it
- [ ] Create git branch (`git checkout -b phase-N-cleanup`)
- [ ] Commit before deletion (`git commit -m "Before deletion"`)
- [ ] Delete file
- [ ] Run verification commands
- [ ] If broken: `git reset --hard` and rethink

**Never delete without verification.**

---

## **What NOT to Delete**

❌ **DO NOT DELETE:**
- Type definitions (`.d.ts` files)
- Test files (`*.test.ts`, `*.spec.ts`)
- Configuration files (`tsconfig.json`, `vite.config.ts`)
- Documentation files (unless replaced)
- Shared utilities (unless proven unused)

✅ **SAFE TO DELETE:**
- `DaemonContainer.tsx` (replaced by QueryContainer)
- `SshHivesContainer.tsx` (replaced by hooks)
- Promise cache logic (replaced by query cache)
- `queueMicrotask` calls (architectural hack)
- `enableMapSet()` calls (no longer needed)
- Duplicate loading/error state (query cache owns it)
