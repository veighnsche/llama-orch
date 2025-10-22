# 📋 WORK UNITS CHECKLIST
## Migration Progress Tracking

**Total Units:** 24  
**BDD Units:** 12  
**Code Units:** 12  
**Status:** 🔴 0/24 Complete  

---

## 🧪 PHASE 1: BDD TEST MIGRATION (12 units)

### ✅ = Complete | 🟡 = In Progress | 🔴 = Not Started

- [ ] **UNIT 1-A** 🔴 rbee-keeper BDD Tests (2-3h)
- [ ] **UNIT 1-B** 🔴 queen-rbee Registry BDD Tests (2-3h)
- [ ] **UNIT 1-C** 🔴 queen-rbee Lifecycle & Preflight BDD (2-3h)
- [ ] **UNIT 1-D** 🔴 rbee-hive Model Management BDD (2-3h)
- [ ] **UNIT 1-E** 🔴 rbee-hive Worker Management BDD (2-3h)
- [ ] **UNIT 1-F** 🔴 rbee-hive Lifecycle & Monitor BDD (2-3h)
- [ ] **UNIT 1-G** 🔴 Worker BDD Tests (2-3h)
- [ ] **UNIT 1-H** 🔴 Shared Crates BDD - Auth & Secrets (2-3h)
- [ ] **UNIT 1-I** 🔴 Shared Crates BDD - Validation & Narration (2-3h)
- [ ] **UNIT 1-J** 🔴 Shared Crates BDD - HTTP & Types (2-3h)
- [ ] **UNIT 1-K** 🔴 Integration & Concurrency BDD (2-3h)
- [ ] **UNIT 1-L** 🔴 End-to-End BDD (2-3h)

**Phase 1 Progress:** 0/12 (0%)

---

## 💻 PHASE 2: CODE MIGRATION (12 units)

- [ ] **UNIT 2-A** 🔴 rbee-keeper Commands (3-4h)
- [ ] **UNIT 2-B** 🔴 rbee-keeper Lifecycle & Config (3-4h)
- [ ] **UNIT 2-C** 🔴 queen-rbee SSH & Registries (3-4h)
- [ ] **UNIT 2-D** 🔴 queen-rbee Lifecycle & Preflight (3-4h)
- [ ] **UNIT 2-E** 🔴 queen-rbee HTTP Server & Scheduler (3-4h)
- [ ] **UNIT 2-F** 🔴 rbee-hive Model Management (3-4h)
- [ ] **UNIT 2-G** 🔴 rbee-hive Worker Management (3-4h)
- [ ] **UNIT 2-H** 🔴 rbee-hive Monitor & Resources (3-4h)
- [ ] **UNIT 2-I** 🔴 rbee-hive HTTP Server & Main (3-4h)
- [ ] **UNIT 2-J** 🔴 Shared Crate - daemon-lifecycle (3-4h)
- [ ] **UNIT 2-K** 🔴 Shared Crate - rbee-http-client (3-4h)
- [ ] **UNIT 2-L** 🔴 Shared Crate - rbee-types (3-4h)

**Phase 2 Progress:** 0/12 (0%)

---

## 👥 TEAM ASSIGNMENTS

### Person A (BDD Specialist)
- [ ] UNIT 1-A
- [ ] UNIT 1-D
- [ ] UNIT 1-G
- [ ] UNIT 1-J
- [ ] UNIT 2-A (with Person C)
- [ ] UNIT 2-B (with Person C)

### Person B (queen-rbee Specialist)
- [ ] UNIT 1-B
- [ ] UNIT 1-C
- [ ] UNIT 2-C
- [ ] UNIT 2-D
- [ ] UNIT 2-E

### Person C (rbee-hive Specialist)
- [ ] UNIT 1-E
- [ ] UNIT 1-F
- [ ] UNIT 2-F
- [ ] UNIT 2-G
- [ ] UNIT 2-H
- [ ] UNIT 2-I
- [ ] UNIT 2-A (with Person A)
- [ ] UNIT 2-B (with Person A)

### All Team (Shared)
- [ ] UNIT 1-H
- [ ] UNIT 1-I
- [ ] UNIT 1-K
- [ ] UNIT 1-L
- [ ] UNIT 2-J
- [ ] UNIT 2-K
- [ ] UNIT 2-L

---

## 🎯 DAILY STANDUP TEMPLATE

**Date:** __________

**Person A Progress:**
- Working on: __________
- Completed today: __________
- Blocked by: __________

**Person B Progress:**
- Working on: __________
- Completed today: __________
- Blocked by: __________

**Person C Progress:**
- Working on: __________
- Completed today: __________
- Blocked by: __________

**Team Decisions:**
- __________

---

## 📊 MILESTONE TRACKING

### Milestone 1: BDD Foundation (Week 1)
**Target:** Complete Units 1-A through 1-L
- [ ] All feature files migrated
- [ ] All step definitions created
- [ ] All BDD tests pass

### Milestone 2: Shared Crates (Week 2)
**Target:** Complete Units 2-J, 2-K, 2-L
- [ ] daemon-lifecycle complete
- [ ] rbee-http-client complete
- [ ] rbee-types complete

### Milestone 3: rbee-keeper (Week 2)
**Target:** Complete Units 2-A, 2-B
- [ ] Commands migrated
- [ ] Lifecycle migrated
- [ ] Tests pass

### Milestone 4: queen-rbee (Week 2-3)
**Target:** Complete Units 2-C, 2-D, 2-E
- [ ] SSH & registries migrated
- [ ] Lifecycle migrated
- [ ] HTTP server migrated
- [ ] Tests pass

### Milestone 5: rbee-hive (Week 3)
**Target:** Complete Units 2-F, 2-G, 2-H, 2-I
- [ ] Model management migrated
- [ ] Worker management migrated
- [ ] Monitor migrated
- [ ] HTTP server migrated
- [ ] Tests pass

### Milestone 6: Integration (Week 4)
**Target:** Full system working
- [ ] All services compile
- [ ] All unit tests pass
- [ ] All BDD tests pass
- [ ] End-to-end flow works

---

## ⚠️ CRITICAL PATH

### Must Complete First (Blocking others):
1. 🔴 UNIT 2-L: rbee-types (blocks all code units)
2. 🔴 UNIT 2-K: rbee-http-client (blocks keeper & queen)
3. 🔴 UNIT 2-J: daemon-lifecycle (blocks keeper & queen)

### Can Work in Parallel:
- All BDD units (1-A through 1-L)
- rbee-keeper units (after shared crates)
- queen-rbee units (after shared crates)
- rbee-hive units (after shared crates)

---

## 🔥 QUICK START

**Day 1 Priorities:**
1. Person A: Start UNIT 1-A (rbee-keeper BDD)
2. Person B: Start UNIT 1-B (queen-rbee registry BDD)
3. Person C: Start UNIT 1-E (rbee-hive worker BDD)

**After BDD basics done:**
1. All team: UNIT 2-L (rbee-types) - 1 day
2. All team: UNIT 2-K (rbee-http-client) - 1 day
3. All team: UNIT 2-J (daemon-lifecycle) - 1 day

**Then split by specialty:**
- Person A: rbee-keeper code
- Person B: queen-rbee code
- Person C: rbee-hive code

---

## 📝 COMPLETION CRITERIA

### Each Unit Complete When:
- [ ] Code migrated and compiles
- [ ] Unit tests written and passing
- [ ] BDD tests written and passing
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Checked off in this list

### Project Complete When:
- [ ] All 24 units complete
- [ ] `cargo build --all` passes
- [ ] `cargo test --all` passes
- [ ] End-to-end flow documented
- [ ] Architecture docs updated
- [ ] Old code can be archived

---

**Last Updated:** 2025-10-20  
**Next Update:** __________
