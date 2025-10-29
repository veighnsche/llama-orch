# 🐛 Critical Bug Fixes (TEAM-355)

**Date:** 2025-10-29  
**Team:** TEAM-355  
**Status:** ✅ FIXED

---

## **Bug 1: Map/Set Requires enableMapSet()** 🔥

### **Error:**
```
[Immer] The plugin for 'MapSet' has not been loaded into Immer.
To enable the plugin, import and call `enableMapSet()` when initializing your application.
```

### **Root Cause:**
We removed `enableMapSet()` import but were still using `Map<string, HiveQuery>` in the store state. Immer requires the MapSet plugin to handle Map/Set in mutable drafts.

### **Why This Happened:**
When we deleted promise caching, we removed `enableMapSet()` but kept `Map` for the query cache. Immer can't handle Map without the plugin.

### **Solution:**
Replace `Map` with plain object (`Record<string, HiveQuery>`).

**Before (Broken):**
```typescript
interface SshHivesState {
  queries: Map<string, HiveQuery>  // ❌ Requires enableMapSet()
}

export const useSshHivesStore = create<SshHivesState>()(
  immer((set, get) => ({
    queries: new Map(),  // ❌ Crashes without enableMapSet()
    
    fetchHive: async (hiveId: string) => {
      const existing = get().queries.get(hiveId)  // ❌ Map API
      set((state) => {
        state.queries.set(hiveId, { ... })  // ❌ Map API
      })
    }
  }))
)
```

**After (Fixed):**
```typescript
interface SshHivesState {
  queries: Record<string, HiveQuery>  // ✅ Plain object, no plugin needed
}

export const useSshHivesStore = create<SshHivesState>()(
  immer((set, get) => ({
    queries: {},  // ✅ Plain object
    
    fetchHive: async (hiveId: string) => {
      const existing = get().queries[hiveId]  // ✅ Object access
      set((state) => {
        state.queries[hiveId] = { ... }  // ✅ Object assignment
      })
    }
  }))
)
```

### **Benefits:**
- ✅ No `enableMapSet()` needed
- ✅ More serializable (plain object)
- ✅ Simpler API (`obj[key]` vs `map.get(key)`)
- ✅ Better TypeScript inference
- ✅ Works with Immer out of the box

### **Changes:**
- `queries: Map<string, HiveQuery>` → `queries: Record<string, HiveQuery>`
- `new Map()` → `{}`
- `queries.get(hiveId)` → `queries[hiveId]`
- `queries.set(hiveId, value)` → `queries[hiveId] = value`
- `queries.delete(hiveId)` → `delete queries[hiveId]`
- `queries.clear()` → `queries = {}`

---

## **Bug 2: Infinite Loop in useEffect** 🔥

### **Error:**
Infinite re-renders when components mount.

### **Root Cause:**
`useEffect` dependency arrays included `store`, which is a Zustand store object that has a **new reference on every render**.

**Why This Happened:**
```typescript
export function useHive(hiveId: string) {
  const store = useSshHivesStore()  // ← New reference every render
  
  useEffect(() => {
    store.fetchHive(hiveId)
  }, [hiveId, store])  // ❌ store changes every render → infinite loop
}
```

### **How the Bug Manifests:**
1. Component renders
2. `useSshHivesStore()` returns store object
3. `useEffect` runs because `store` reference changed
4. `fetchHive` updates state
5. Component re-renders
6. Go to step 2 → **INFINITE LOOP**

### **Solution:**
Extract the function from the store and use it in deps instead.

**Before (Broken):**
```typescript
export function useHive(hiveId: string) {
  const store = useSshHivesStore()
  const query = store.queries[hiveId]
  
  useEffect(() => {
    store.fetchHive(hiveId)
  }, [hiveId, store])  // ❌ Infinite loop
  
  return { hive: query?.data ?? null, ... }
}
```

**After (Fixed):**
```typescript
export function useHive(hiveId: string) {
  const store = useSshHivesStore()
  const query = store.queries[hiveId]
  const fetchHive = store.fetchHive  // ✅ Extract function
  
  useEffect(() => {
    fetchHive(hiveId)
  }, [hiveId, fetchHive])  // ✅ Function is stable (Zustand guarantees this)
  
  return { hive: query?.data ?? null, ... }
}
```

### **Why This Works:**
Zustand guarantees that **store methods are stable** - they don't change between renders. By extracting `fetchHive` separately, we get a stable reference that won't trigger infinite loops.

### **Files Fixed:**
- `src/store/hiveStore.ts` - `useHive()` and `useSshHives()`
- `src/store/queenStore.ts` - `useQueen()`

---

## **Verification**

### **Bug 1 (Map/Set):**
```bash
# Before: Crashes with Immer MapSet error
# After: Works perfectly

grep -r "new Map" src/  # ✅ 0 results
grep -r "Map<" src/     # ✅ 0 results
grep -r "enableMapSet" src/  # ✅ 0 results
```

### **Bug 2 (Infinite Loop):**
```bash
# Before: Components re-render infinitely
# After: Components render once, fetch once

# Check no store in useEffect deps:
grep -A 3 "useEffect" src/store/hiveStore.ts
# Should see: [hiveId, fetchHive] or [fetchHivesList]
# Should NOT see: [hiveId, store] or [store]
```

---

## **Edge Cases Handled**

### **1. Multiple Components Same ID**
- **Scenario:** Two HiveCards mount for same `hiveId`
- **Behavior:** Both call `fetchHive(id)`, but deduplication prevents double fetch
- **Code:** `if (existing?.isLoading) return` in `fetchHive`

### **2. Component Unmount During Fetch**
- **Scenario:** Component unmounts while fetch is in progress
- **Behavior:** Fetch completes, state updates, but component is gone (no memory leak)
- **Code:** Zustand handles this automatically

### **3. Rapid Re-mounts**
- **Scenario:** Component mounts, unmounts, mounts quickly
- **Behavior:** Fresh data check prevents redundant fetches (< 5s)
- **Code:** `if (!force && ... && (now - lastFetch < 5000)) return`

---

## **Summary**

### **Bugs Fixed:**
1. ✅ **Map/Set Immer Error** - Replaced Map with Record
2. ✅ **Infinite Loop** - Extracted functions from store in useEffect deps

### **Files Modified:**
- `src/store/hiveStore.ts` - Map → Record, fixed useEffect deps
- `src/store/queenStore.ts` - Fixed useEffect deps

### **Impact:**
- **Before:** App crashes on load (Immer error) + infinite loops
- **After:** App works perfectly, no crashes, no infinite loops

---

## **Lessons Learned**

### **1. Map/Set in Immer:**
- ❌ **Don't:** Use Map/Set in Immer without `enableMapSet()`
- ✅ **Do:** Use plain objects (`Record<K, V>`) instead
- **Why:** Simpler, more serializable, no plugin needed

### **2. useEffect Dependencies:**
- ❌ **Don't:** Put entire store object in deps
- ✅ **Do:** Extract specific functions from store
- **Why:** Store reference changes every render, functions are stable

### **3. Zustand Guarantees:**
- Store methods are **stable** (don't change between renders)
- Safe to use in useEffect deps when extracted separately
- Entire store object is **not stable** (new reference every render)

---

**All critical bugs fixed. App now stable.**
