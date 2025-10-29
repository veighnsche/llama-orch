# 📚 REFACTOR DOCUMENTATION INDEX

**Last Updated:** 2025-10-29  
**Status:** Ready for implementation

---

## **🚀 Quick Start**

**New developer?** Start here:

1. Read [START_HERE.md](./START_HERE.md) - Overview and team assignments
2. Read [ARCHITECTURAL_BLUNDERS.md](./ARCHITECTURAL_BLUNDERS.md) - What's broken
3. Read [CORRECT_ARCHITECTURE.md](./CORRECT_ARCHITECTURE.md) - The solution
4. Read your assigned phase file

**Don't skip the blunders analysis.** You need to understand WHY the current code is broken.

---

## **📋 Documents by Purpose**

### **Understanding the Problem**

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [ARCHITECTURAL_BLUNDERS.md](./ARCHITECTURAL_BLUNDERS.md) | Complete analysis of what's broken | 20 min |
| [REMOVAL_PLAN.md](./REMOVAL_PLAN.md) | What code to delete and why | 10 min |

### **Understanding the Solution**

| Document | Purpose | Read Time |
|----------|---------|-----------|
| [CORRECT_ARCHITECTURE.md](./CORRECT_ARCHITECTURE.md) | Query-based pattern explained | 15 min |
| [MASTER_PLAN.md](./MASTER_PLAN.md) | Strategy and decision log | 10 min |

### **Implementation Guides**

| Document | Purpose | Duration | Team |
|----------|---------|----------|------|
| [PHASE_1_FIX_LOCALHOST.md](./PHASE_1_FIX_LOCALHOST.md) | Separate localhost from SSH | 1 day | TEAM-350 |
| [PHASE_2_QUERY_STORES.md](./PHASE_2_QUERY_STORES.md) | Implement query pattern | 2 days | TEAM-351 |
| [PHASE_3_SIMPLIFY_CONTAINERS.md](./PHASE_3_SIMPLIFY_CONTAINERS.md) | Create QueryContainer | 1 day | TEAM-352 |
| [PHASE_4_CLEANUP.md](./PHASE_4_CLEANUP.md) | Remove hacks, polish | 1 day | TEAM-353 |

---

## **📖 Reading Order by Role**

### **Team Lead**

Read all documents in this order:

1. START_HERE.md (5 min) - Overview
2. ARCHITECTURAL_BLUNDERS.md (20 min) - Problems
3. CORRECT_ARCHITECTURE.md (15 min) - Solution
4. MASTER_PLAN.md (10 min) - Strategy
5. All 4 phase files (30 min) - Implementation details

**Total:** ~90 minutes

### **Developer (TEAM-350 - Phase 1)**

1. START_HERE.md (5 min)
2. ARCHITECTURAL_BLUNDERS.md - Focus on Blunder #2 (5 min)
3. CORRECT_ARCHITECTURE.md - Focus on Pattern #5 (5 min)
4. PHASE_1_FIX_LOCALHOST.md (15 min)
5. REMOVAL_PLAN.md - Focus on Phase 1 section (5 min)

**Total:** ~35 minutes

### **Developer (TEAM-351 - Phase 2)**

1. START_HERE.md (5 min)
2. ARCHITECTURAL_BLUNDERS.md - Focus on Blunders #3, #4 (10 min)
3. CORRECT_ARCHITECTURE.md - Focus on Patterns #1, #2 (10 min)
4. PHASE_2_QUERY_STORES.md (20 min)
5. REMOVAL_PLAN.md - Focus on Phase 2 section (5 min)

**Total:** ~50 minutes

### **Developer (TEAM-352 - Phase 3)**

1. START_HERE.md (5 min)
2. ARCHITECTURAL_BLUNDERS.md - Focus on Blunders #1, #5, #6 (10 min)
3. CORRECT_ARCHITECTURE.md - Focus on Pattern #3, #4 (10 min)
4. PHASE_3_SIMPLIFY_CONTAINERS.md (20 min)
5. REMOVAL_PLAN.md - Focus on Phase 3 section (5 min)

**Total:** ~50 minutes

### **Developer (TEAM-353 - Phase 4)**

1. START_HERE.md (5 min)
2. All previous phase files (quick scan, 20 min)
3. PHASE_4_CLEANUP.md (20 min)
4. REMOVAL_PLAN.md - Focus on Phase 4 section (5 min)

**Total:** ~50 minutes

---

## **🎯 Documents by Topic**

### **Localhost Issues**

- ARCHITECTURAL_BLUNDERS.md - Blunder #2
- PHASE_1_FIX_LOCALHOST.md - Full implementation
- CORRECT_ARCHITECTURE.md - Pattern #5

### **Promise Caching Issues**

- ARCHITECTURAL_BLUNDERS.md - Blunder #3
- PHASE_2_QUERY_STORES.md - Query pattern solution
- REMOVAL_PLAN.md - Phase 2 deletions

### **Container Issues**

- ARCHITECTURAL_BLUNDERS.md - Blunders #1, #5, #6
- PHASE_3_SIMPLIFY_CONTAINERS.md - QueryContainer solution
- REMOVAL_PLAN.md - Phase 3 deletions

### **Query Pattern**

- CORRECT_ARCHITECTURE.md - All patterns
- PHASE_2_QUERY_STORES.md - Implementation
- MASTER_PLAN.md - Decision log

---

## **📊 Key Metrics**

### **Code Reduction**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Total LOC | ~1000 | ~600 | **40% reduction** |
| Container complexity | 161 LOC | 40 LOC | **75% simpler** |
| Promise hacks | 4 files | 0 files | **100% removed** |

### **Time Investment**

| Phase | Duration | Team | Deliverables |
|-------|----------|------|--------------|
| Phase 1 | 1 day | TEAM-350 | Localhost separation |
| Phase 2 | 2 days | TEAM-351 | Query pattern |
| Phase 3 | 1 day | TEAM-352 | QueryContainer |
| Phase 4 | 1 day | TEAM-353 | Cleanup |
| **Total** | **5 days** | **4 teams** | **Clean architecture** |

### **Quality Improvement**

| Metric | Before | After |
|--------|--------|-------|
| Type safety | Partial | Full ✅ |
| Deduplication | Broken ❌ | Works ✅ |
| Race conditions | Yes ❌ | No ✅ |
| Promise hacks | Yes ❌ | No ✅ |

---

## **🔗 Cross-References**

### **Problem → Solution Mapping**

| Problem (Blunder) | Solution (Phase) | Pattern |
|-------------------|------------------|---------|
| Localhost doesn't work | Phase 1 | LocalhostHive component |
| Promise caching | Phase 2 | Query cache |
| Broken deduplication | Phase 2 | Query hooks |
| Container confusion | Phase 3 | QueryContainer |
| Duplicate state | Phase 4 | Single source of truth |
| Backwards data flow | Phase 2 + 3 | Hooks + Container |

### **File → Phase Mapping**

| File | Phase | Action |
|------|-------|--------|
| `src/store/hiveStore.ts` | Phase 2 | Rewrite with query pattern |
| `src/store/queenStore.ts` | Phase 2 | Rewrite with query pattern |
| `src/containers/DaemonContainer.tsx` | Phase 3 | Delete (Rule Zero) |
| `src/containers/QueryContainer.tsx` | Phase 3 | Create new |
| `src/components/cards/HiveCard.tsx` | Phase 3 | Rewrite with QueryContainer |
| `src/components/cards/LocalhostHive.tsx` | Phase 1 | Create new |
| `src/components/InstalledHiveList.tsx` | Phase 1 + 3 | Modify then rewrite |

---

## **❓ FAQ**

### **Q: Can we skip Phase 1 and go straight to Phase 2?**

**A:** No. Phase 1 is foundational. Localhost must be separated from SSH logic before implementing the query pattern. Skipping it will cause confusion and bugs.

### **Q: Can we patch the existing code instead of rewriting?**

**A:** No. The architecture is fundamentally broken (see ARCHITECTURAL_BLUNDERS.md). Patches will just add more hacks. Rule Zero: Delete complexity, don't add layers.

### **Q: Why not use React Query library instead of custom pattern?**

**A:** Good question. We considered it. Decision: Custom pattern is simpler for this use case (3 queries total). React Query is overkill. See MASTER_PLAN.md decision log.

### **Q: What if we find more issues during implementation?**

**A:** Document them in phase files. If it's a new blunder, add to ARCHITECTURAL_BLUNDERS.md. If it changes the plan, update MASTER_PLAN.md decision log.

### **Q: How do we handle merge conflicts?**

**A:** Each phase is a separate branch. Merge in order: Phase 1 → Phase 2 → Phase 3 → Phase 4. Don't work on multiple phases in parallel.

### **Q: What if tests fail after refactor?**

**A:** Fix them. Don't merge broken code. Tests might need updating to match new patterns. See phase files for guidance.

---

## **✅ Checklist for Completion**

### **Documentation**

- [x] START_HERE.md created
- [x] ARCHITECTURAL_BLUNDERS.md created
- [x] CORRECT_ARCHITECTURE.md created
- [x] MASTER_PLAN.md created
- [x] REMOVAL_PLAN.md created
- [x] INDEX.md created (this file)
- [x] All 4 phase files created

### **Implementation** (To be checked by teams)

- [ ] Phase 1 complete (TEAM-350)
- [ ] Phase 2 complete (TEAM-351)
- [ ] Phase 3 complete (TEAM-352)
- [ ] Phase 4 complete (TEAM-353)

### **Verification** (To be checked by team lead)

- [ ] All tests pass
- [ ] TypeScript compiles (strict mode)
- [ ] ESLint clean
- [ ] No queueMicrotask in codebase
- [ ] No enableMapSet in codebase
- [ ] No DaemonContainer references
- [ ] Localhost works without installation
- [ ] SSH hives work with installation
- [ ] Performance metrics good

---

## **🎓 Learning Resources**

### **External References**

- [React Query Docs](https://tanstack.com/query/latest) - Inspiration for query pattern
- [Zustand Best Practices](https://github.com/pmndrs/zustand) - Store patterns
- [TypeScript Generics](https://www.typescriptlang.org/docs/handbook/2/generics.html) - For QueryContainer<T>

### **Internal References**

- `/.windsurf/rules/engineering-rules.md` - Project rules
- `/bin/00_rbee_keeper/ui/README.md` - App overview

---

**Questions?** Read the docs first. Still stuck? Ask team lead.

**Ready to start?** Go to [START_HERE.md](./START_HERE.md)
