# 📋 MASTER PLAN - Container/Store Refactor

**Version:** 1.0  
**Date:** 2025-10-29  
**Estimated Duration:** 5-6 days

---

## **Goals**

1. ✅ Fix localhost hive (frontend/backend alignment)
2. ✅ Remove promise caching from Zustand
3. ✅ Implement query-based pattern (React Query style)
4. ✅ Simplify containers (dumb UI only)
5. ✅ Reduce code by 40% (delete complexity)

---

## **Phases Overview**

| Phase | Duration | Team | Focus |
|-------|----------|------|-------|
| 1 | 1 day | TEAM-350 | Fix Localhost |
| 2 | 2 days | TEAM-351 | Query Stores |
| 3 | 1 day | TEAM-352 | Containers |
| 4 | 1 day | TEAM-353 | Cleanup |

---

## **Phase 1: Fix Localhost** (1 day)

**Goal:** Separate localhost from SSH hives

**Deliverables:**
- `LocalhostHive.tsx` component
- Remove localhost from SSH hive lists
- Backend verification

**Success Criteria:**
- Localhost hive works (start/stop, no install)
- SSH hive list doesn't include localhost
- Tests pass

---

## **Phase 2: Query-Based Stores** (2 days)

**Goal:** Replace promise caching with query cache

**Deliverables:**
- `useHive(id)` hook
- `useSshHives()` hook  
- `useQueen()` hook
- Query cache in stores

**Success Criteria:**
- No `queueMicrotask` hacks
- No `_fetchPromises` in state
- Automatic deduplication works
- Tests pass

---

## **Phase 3: Simplify Containers** (1 day)

**Goal:** Create generic QueryContainer, delete DaemonContainer

**Deliverables:**
- `QueryContainer<T>` component
- Delete `DaemonContainer.tsx` (Rule Zero)
- Update all cards

**Success Criteria:**
- Type-safe containers
- No ErrorBoundary needed
- 40% less code
- Tests pass

---

## **Phase 4: Cleanup** (1 day)

**Goal:** Remove all hacks and dead code

**Deliverables:**
- Remove `isLoading`/`error` from stores
- Remove Immer `enableMapSet`
- Update documentation
- Final testing

**Success Criteria:**
- No TODO markers
- No dead code
- All tests pass
- Documentation updated

---

## **Decision Log**

### **Why Query Pattern?**

**Rejected:** Fix existing DaemonContainer  
**Chosen:** Replace with Query pattern  
**Reason:** Existing architecture is fundamentally broken, patches won't help

### **Why Separate Localhost?**

**Rejected:** Make localhost work as SSH target  
**Chosen:** Separate component for localhost  
**Reason:** Backend treats localhost specially, frontend should too

### **Why Delete DaemonContainer?**

**Rejected:** Make DaemonContainer generic  
**Chosen:** Replace with simpler QueryContainer  
**Reason:** Rule Zero - delete complexity, don't add layers

---

## **Risk Mitigation**

| Risk | Mitigation |
|------|------------|
| Breaking existing behavior | Comprehensive tests before/after |
| Team confusion | Detailed phase guides + examples |
| Scope creep | Strict phase boundaries |
| Rollback needed | Git branch per phase |

---

## **Success Metrics**

- ✅ Localhost hive works without installation
- ✅ No race conditions (verified by tests)
- ✅ No `queueMicrotask` in codebase
- ✅ 40% code reduction
- ✅ Type-safe container pattern
- ✅ All existing tests pass
