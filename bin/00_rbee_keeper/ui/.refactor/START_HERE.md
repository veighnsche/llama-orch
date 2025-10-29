# 🚨 CONTAINER & STORE ARCHITECTURE REFACTOR

**Status:** 🔴 CRITICAL - Current architecture is broken  
**Estimated Effort:** 5-6 days  
**Teams Required:** 4 teams (1 per phase + 1 for review)

---

## **Quick Context**

The current Container/Zustand architecture has fundamental design flaws:

1. ❌ **Localhost hive doesn't work** (frontend/backend mismatch)
2. ❌ **Promise caching in Zustand** (React anti-pattern)
3. ❌ **Double data fetching** (race conditions)
4. ❌ **Container/Store responsibility confusion** (who owns loading state?)
5. ❌ **Global promise cache pollution** (cross-daemon contamination)

**Result:** Broken UX, race conditions, stale data, confusing code

---

## **Solution Summary**

**Replace current architecture with Query-based pattern:**

```
OLD (Broken):
Page → DaemonContainer → fetchFn → Store → Components read store
     ↓                                    ↓
   Loading/Error                      Promise cache hacks

NEW (Correct):
Component → useQuery hook → Store query cache → QueryContainer → Render
          ↓                 ↓                    ↓
        Auto-fetch      Deduplication      Loading/Error UI
```

**Key Changes:**
- ✅ **Stores own queries** - Not containers
- ✅ **Hooks drive fetching** - useEffect in hook, not component
- ✅ **Containers are dumb** - Just UI for loading/error states
- ✅ **Localhost is special** - Not an SSH target
- ✅ **No promise caching** - Queries are data, not promises

---

## **What You'll Do**

### **Phase 1: Fix Localhost** (1 day)
- Create `LocalhostHive` component (no install/uninstall)
- Remove localhost from SSH hive list
- Backend verification (ensure localhost works without hives.conf)

### **Phase 2: Query-Based Stores** (2 days)
- Create `useHive(id)`, `useSshHives()`, `useQueen()` hooks
- Implement query cache pattern (Map<id, {data, loading, error}>)
- Remove promise caching hacks

### **Phase 3: Simplify Containers** (1 day)
- Create generic `QueryContainer<T>`
- Delete `DaemonContainer` (Rule Zero)
- Update all cards to use new pattern

### **Phase 4: Cleanup** (1 day)
- Remove `isLoading`/`error` from stores
- Remove `_fetchPromises`, `queueMicrotask` hacks
- Remove Immer `enableMapSet`

---

## **Reading Order**

1. **[ARCHITECTURAL_BLUNDERS.md](./ARCHITECTURAL_BLUNDERS.md)** - What's broken and why
2. **[CORRECT_ARCHITECTURE.md](./CORRECT_ARCHITECTURE.md)** - The right way to do it
3. **[MASTER_PLAN.md](./MASTER_PLAN.md)** - Strategy and decision log
4. **Phase files** - Step-by-step implementation

---

## **Team Assignments**

| Team | Phase | Duration | Deliverables |
|------|-------|----------|--------------|
| TEAM-350 | Phase 1: Fix Localhost | 1 day | LocalhostHive component, backend fixes |
| TEAM-351 | Phase 2: Query Stores | 2 days | useHive/useSshHives/useQueen hooks |
| TEAM-352 | Phase 3: Containers | 1 day | QueryContainer, delete DaemonContainer |
| TEAM-353 | Phase 4: Cleanup | 1 day | Remove hacks, documentation |

---

## **Success Criteria**

✅ **Localhost hive works** (start/stop without installation)  
✅ **No race conditions** (multiple mounts = 1 fetch)  
✅ **No promise hacks** (`queueMicrotask`, `_fetchPromises` gone)  
✅ **Type safety** (QueryContainer enforces data type)  
✅ **40% less code** (simpler, easier to maintain)  
✅ **All tests pass** (existing behavior preserved)

---

## **Rules**

🔥 **RULE ZERO:** Delete complexity, don't add compatibility layers  
📋 **Read docs first:** Understand the problem before coding  
✅ **Complete previous phase:** Don't start Phase 2 until Phase 1 is done  
🧪 **Test as you go:** Each phase must have working code  
📝 **Update docs:** Mark completed items in phase files

---

## **Quick Start**

```bash
# 1. Read the blunders analysis
cat bin/00_rbee_keeper/ui/.refactor/ARCHITECTURAL_BLUNDERS.md

# 2. Read the correct architecture
cat bin/00_rbee_keeper/ui/.refactor/CORRECT_ARCHITECTURE.md

# 3. Read the master plan
cat bin/00_rbee_keeper/ui/.refactor/MASTER_PLAN.md

# 4. Start Phase 1
cat bin/00_rbee_keeper/ui/.refactor/PHASE_1_FIX_LOCALHOST.md
```

---

**Questions?** Read the docs first. Still stuck? Check master plan decision log.

**Don't skip phases.** This is architectural work - order matters.
