# Phase 2: Query-Based Stores (2 days)

**Team:** TEAM-351  
**Duration:** 2 days  
**Status:** 🔴 NOT STARTED  
**Dependencies:** Phase 1 must be complete

---

## **Goal**

Replace promise caching with query-based pattern. Stores own data + metadata, not promises.

---

## **Background**

**Current (Broken):**
```typescript
// Stores promises in state (anti-pattern)
_fetchPromises: Map<string, Promise<void>>

// Requires hacks
queueMicrotask(() => { set(...) })
enableMapSet()
```

**Target:**
```typescript
// Stores queries (data + metadata)
queries: Map<string, { data, isLoading, error, lastFetch }>

// No hacks needed
// Automatic deduplication
// Stale-while-revalidate
```

---

## **Tasks**

### **Task 1: Create Query-Based Hive Store** (4 hours)

**File:** `src/store/hiveStore.ts` (REWRITE)

```typescript
// TEAM-351: Query-based hive store (replaces promise caching)
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { commands } from '@/generated/bindings'

interface SshHive {
  host: string
  hostname: string
  user: string
  port: number
  status: 'online' | 'offline' | 'unknown'
  isInstalled: boolean
}

interface HiveQuery {
  data: SshHive | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}

interface HiveStoreState {
  // Query cache - one entry per hive
  queries: Map<string, HiveQuery>
  
  // SSH hive list (from ~/.ssh/config)
  sshHives: SshHive[]
  sshHivesLoading: boolean
  sshHivesError: string | null
  
  // Actions
  fetchHive: (hiveId: string, force?: boolean) => Promise<void>
  fetchSshHives: () => Promise<void>
  invalidate: (hiveId: string) => void
  invalidateAll: () => void
}

export const useHiveStore = create<HiveStoreState>()(
  immer((set, get) => ({
    queries: new Map(),
    sshHives: [],
    sshHivesLoading: false,
    sshHivesError: null,
    
    fetchHive: async (hiveId: string, force = false) => {
      const now = Date.now()
      const existing = get().queries.get(hiveId)
      
      // Stale-while-revalidate: Skip if fresh (< 5s old)
      if (!force && existing && !existing.isLoading && (now - existing.lastFetch < 5000)) {
        return
      }
      
      // Deduplication: Skip if already loading
      if (existing?.isLoading) {
        return
      }
      
      // Start loading
      set((state) => {
        state.queries.set(hiveId, {
          data: existing?.data ?? null,
          isLoading: true,
          error: null,
          lastFetch: now,
        })
      })
      
      try {
        const result = await commands.hiveStatus(hiveId)
        if (result.status === 'ok') {
          const hive: SshHive = {
            host: hiveId,
            hostname: result.data.hostname,
            user: result.data.user,
            port: result.data.port,
            status: result.data.is_running ? 'online' : 'offline',
            isInstalled: result.data.is_installed,
          }
          set((state) => {
            state.queries.set(hiveId, {
              data: hive,
              isLoading: false,
              error: null,
              lastFetch: now,
            })
          })
        } else {
          throw new Error(result.error || 'Failed to fetch hive status')
        }
      } catch (error) {
        set((state) => {
          state.queries.set(hiveId, {
            data: existing?.data ?? null,
            isLoading: false,
            error: error instanceof Error ? error.message : 'Failed to fetch hive',
            lastFetch: now,
          })
        })
      }
    },
    
    fetchSshHives: async () => {
      if (get().sshHivesLoading) return
      
      set((state) => {
        state.sshHivesLoading = true
        state.sshHivesError = null
      })
      
      try {
        const result = await commands.sshList()
        if (result.status === 'ok') {
          set((state) => {
            state.sshHives = result.data.map(convertToSshHive)
            state.sshHivesLoading = false
          })
        } else {
          throw new Error(result.error || 'Failed to load SSH hives')
        }
      } catch (error) {
        set((state) => {
          state.sshHivesLoading = false
          state.sshHivesError = error instanceof Error ? error.message : 'Failed'
        })
      }
    },
    
    invalidate: (hiveId: string) => {
      set((state) => { state.queries.delete(hiveId) })
    },
    
    invalidateAll: () => {
      set((state) => { state.queries.clear() })
    },
  }))
)
```

---

### **Task 2: Create useHive Hook** (1 hour)

**File:** `src/hooks/useHive.ts`

```typescript
// TEAM-351: Query hook for individual hive status
import { useEffect } from 'react'
import { useHiveStore } from '@/store/hiveStore'
import { useHiveActions } from './useHiveActions'

export function useHive(hiveId: string) {
  const store = useHiveStore()
  const query = store.queries.get(hiveId)
  const actions = useHiveActions()
  
  // Fetch on mount (automatic)
  useEffect(() => {
    store.fetchHive(hiveId)
  }, [hiveId, store])
  
  return {
    hive: query?.data ?? null,
    isLoading: query?.isLoading ?? true,
    error: query?.error ?? null,
    refetch: () => store.fetchHive(hiveId, true),
    ...actions,  // start, stop, install, etc.
  }
}
```

---

### **Task 3: Create useSshHives Hook** (1 hour)

**File:** `src/hooks/useSshHives.ts`

```typescript
// TEAM-351: Query hook for SSH hive list
import { useEffect } from 'react'
import { useHiveStore } from '@/store/hiveStore'

export function useSshHives() {
  const store = useHiveStore()
  
  useEffect(() => {
    store.fetchSshHives()
  }, [store])
  
  return {
    hives: store.sshHives,
    isLoading: store.sshHivesLoading,
    error: store.sshHivesError,
    refetch: () => store.fetchSshHives(),
  }
}
```

---

### **Task 4: Create Query-Based Queen Store** (3 hours)

**File:** `src/store/queenStore.ts` (REWRITE)

```typescript
// TEAM-351: Query-based queen store
import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'

interface QueenStatus {
  isRunning: boolean
  isInstalled: boolean
}

interface QueenQuery {
  data: QueenStatus | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}

interface QueenStoreState {
  query: QueenQuery
  fetchQueen: (force?: boolean) => Promise<void>
  invalidate: () => void
}

export const useQueenStore = create<QueenStoreState>()(
  immer((set, get) => ({
    query: {
      data: null,
      isLoading: false,
      error: null,
      lastFetch: 0,
    },
    
    fetchQueen: async (force = false) => {
      const now = Date.now()
      const { query } = get()
      
      // Skip if fresh
      if (!force && !query.isLoading && (now - query.lastFetch < 5000)) {
        return
      }
      
      // Skip if loading
      if (query.isLoading) return
      
      set((state) => {
        state.query.isLoading = true
        state.query.error = null
        state.query.lastFetch = now
      })
      
      try {
        const result = await commands.queenStatus()
        if (result.status === 'ok') {
          set((state) => {
            state.query.data = {
              isRunning: result.data.is_running,
              isInstalled: result.data.is_installed,
            }
            state.query.isLoading = false
          })
        } else {
          throw new Error(result.error)
        }
      } catch (error) {
        set((state) => {
          state.query.isLoading = false
          state.query.error = error instanceof Error ? error.message : 'Failed'
        })
      }
    },
    
    invalidate: () => {
      set((state) => {
        state.query = { data: null, isLoading: false, error: null, lastFetch: 0 }
      })
    },
  }))
)
```

---

### **Task 5: Create useQueen Hook** (30 min)

**File:** `src/hooks/useQueen.ts`

```typescript
// TEAM-351: Query hook for queen status
import { useEffect } from 'react'
import { useQueenStore } from '@/store/queenStore'
import { useQueenActions } from './useQueenActions'

export function useQueen() {
  const store = useQueenStore()
  const actions = useQueenActions()
  
  useEffect(() => {
    store.fetchQueen()
  }, [store])
  
  return {
    queen: store.query.data,
    isLoading: store.query.isLoading,
    error: store.query.error,
    refetch: () => store.fetchQueen(true),
    ...actions,
  }
}
```

---

## **Checklist**

- [ ] Rewrite `hiveStore.ts` with query pattern
- [ ] Create `useHive(id)` hook
- [ ] Create `useSshHives()` hook
- [ ] Rewrite `queenStore.ts` with query pattern
- [ ] Create `useQueen()` hook
- [ ] Remove ALL `queueMicrotask` calls
- [ ] Remove `_fetchPromises` from stores
- [ ] Remove `enableMapSet()` from stores
- [ ] Test automatic deduplication (mount 3 cards, verify 1 fetch)
- [ ] Test stale-while-revalidate (old data shown while fetching)
- [ ] All tests pass

---

## **Success Criteria**

✅ No `queueMicrotask` in codebase  
✅ No `_fetchPromises` in state  
✅ No `enableMapSet()` needed  
✅ Multiple components can call same hook safely  
✅ Stale data shown while refetching (better UX)  
✅ Type-safe query state

---

## **Testing**

```typescript
// Test deduplication
function TestPage() {
  return (
    <>
      <HiveCard hiveId="localhost" />  {/* Uses useHive("localhost") */}
      <HiveCard hiveId="localhost" />  {/* Same hook, should not double-fetch */}
      <HiveCard hiveId="localhost" />  {/* Same hook, should not triple-fetch */}
    </>
  )
}
// Expected: Only 1 fetch for "localhost"
```
