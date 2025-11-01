# UI Implementation Roadmap

**Created:** October 29, 2025  
**Based On:** TEAM-350 Learnings  
**Total Teams:** 5 (TEAM-351 through TEAM-355)  
**Estimated Duration:** 8-10 days total

---

## Overview

This roadmap breaks down the complete UI implementation into 5 distinct phases. Each phase has its own comprehensive document with step-by-step instructions.

**Goal:** Implement Queen, Hive, and Worker UIs with zero code duplication using shared packages.

---

## Phase Breakdown

### 🏗️ TEAM-351: Shared Packages Creation (2-3 days)

**Status:** 🔜 TODO  
**Priority:** CRITICAL (Blocks all other phases)  
**Document:** `TEAM_351_SHARED_PACKAGES_PHASE.md`

**Mission:** Create 4 reusable packages that eliminate duplication.

**Deliverables:**
- `@rbee/shared-config` - Port configuration (single source of truth)
- `@rbee/narration-client` - Narration SSE handling
- `@rbee/iframe-bridge` - iframe ↔ parent communication
- `@rbee/dev-utils` - Environment detection utilities
- Rust constants generator

**Why First:** All other phases depend on these packages.

---

### ✅ TEAM-352: Queen UI Migration (1 day)

**Status:** 🔜 TODO  
**Priority:** HIGH  
**Dependencies:** TEAM-351  
**Document:** `TEAM_352_QUEEN_MIGRATION_PHASE.md`

**Mission:** Migrate Queen UI from duplicate code to shared packages.

**Deliverables:**
- Queen UI uses `@rbee/narration-client`
- Keeper uses `@rbee/shared-config` for iframe URLs
- Keeper uses `@rbee/shared-config` for allowed origins
- ~110 LOC removed (85% reduction)
- Pattern validated

**Why Second:** Validates shared packages work before Hive/Worker use them.

---

### 🏗️ TEAM-353: Hive UI Implementation (2-3 days)

**Status:** 🔜 TODO  
**Priority:** HIGH  
**Dependencies:** TEAM-351, TEAM-352  
**Document:** `TEAM_353_HIVE_UI_PHASE.md`

**Mission:** Implement Hive UI from scratch using shared packages.

**Deliverables:**
- Hive WASM SDK package
- Hive React hooks package
- Hive Vite app
- build.rs for Hive
- Keeper HivePage
- ~120 LOC saved (96% reduction vs duplicating)

**Why Third:** First new service using proven pattern.

---

### 🏗️ TEAM-354: Worker UI Implementation (2-3 days)

**Status:** 🔜 TODO  
**Priority:** MEDIUM  
**Dependencies:** TEAM-351, TEAM-352, TEAM-353  
**Document:** `TEAM_354_WORKER_UI_PHASE.md`

**Mission:** Implement Worker UI using proven pattern.

**Deliverables:**
- Worker WASM SDK package
- Worker React hooks package
- Worker Vite app
- build.rs for Worker
- Keeper WorkerPage
- Complete UI suite (Keeper, Queen, Hive, Worker)

**Why Fourth:** Final service completes the suite.

---

### 📚 TEAM-355: Final Documentation (1 day)

**Status:** 🔜 TODO  
**Priority:** LOW  
**Dependencies:** TEAM-351, 352, 353, 354  
**Document:** `TEAM_355_FINAL_DOCUMENTATION_PHASE.md`

**Mission:** Create comprehensive documentation and close out project.

**Deliverables:**
- Updated PORT_CONFIGURATION.md
- Architecture diagrams
- Quick start guide
- Shared package documentation
- TEAM-350 archive
- Final handoff summary

**Why Last:** Documents the complete implementation.

---

## Quick Reference

### Port Allocation

| Service | Dev  | Prod | Backend |
|---------|------|------|---------|
| Keeper  | 5173 | Tauri| N/A     |
| Queen   | 7834 | 7833 | 7833    |
| Hive    | 7836 | 7835 | 7835    |
| Worker  | 7837 | 8080 | 8080    |

### Shared Packages

1. **@rbee/shared-config** - Port management
2. **@rbee/narration-client** - Narration handling
3. **@rbee/iframe-bridge** - Communication
4. **@rbee/dev-utils** - Utilities

### Code Savings

- **Per Service:** ~120 LOC saved
- **3 Services:** ~360 LOC total saved
- **Maintenance:** 40% reduction

---

## Team Instructions

### For Each Team

1. **Read your phase document** - Comprehensive step-by-step guide
2. **Read prerequisite docs** - Listed in your phase document
3. **Follow the checklist** - Every deliverable listed
4. **Test both modes** - Development and production
5. **Create handoff doc** - Summary for next team

### Critical Rules

🚨 **Never hardcode ports** - Use `@rbee/shared-config`  
🚨 **Never duplicate code** - Use shared packages  
🚨 **Test both modes** - Dev and prod must work  
🚨 **Update all locations** - When adding services

---

## Progress Tracking

### TEAM-351: Shared Packages
- [ ] @rbee/shared-config created
- [ ] @rbee/narration-client created
- [ ] @rbee/iframe-bridge created
- [ ] @rbee/dev-utils created
- [ ] Rust constants generator working
- [ ] All packages in pnpm workspace

### TEAM-352: Queen Migration
- [ ] Queen uses shared packages
- [ ] Keeper uses shared config
- [ ] Duplicate code removed
- [ ] Both modes tested
- [ ] Pattern validated

### TEAM-353: Hive UI
- [ ] Hive SDK created
- [ ] Hive React created
- [ ] Hive App created
- [ ] build.rs configured
- [ ] Keeper HivePage added
- [ ] Both modes tested

### TEAM-354: Worker UI
- [ ] Worker SDK created
- [ ] Worker React created
- [ ] Worker App created
- [ ] build.rs configured
- [ ] Keeper WorkerPage added
- [ ] All 4 services working

### TEAM-355: Documentation
- [ ] PORT_CONFIGURATION.md updated
- [ ] Architecture diagrams created
- [ ] Quick start guide written
- [ ] Shared packages documented
- [ ] TEAM-350 archived
- [ ] Final summary complete

---

## Timeline

```
Week 1:
  Day 1-3: TEAM-351 (Shared Packages)
  Day 4:   TEAM-352 (Queen Migration)
  Day 5:   TEAM-353 start (Hive UI)

Week 2:
  Day 1-2: TEAM-353 finish (Hive UI)
  Day 3-4: TEAM-354 (Worker UI)
  Day 5:   TEAM-355 (Documentation)
```

**Total:** 8-10 working days

---

## Success Metrics

### Code Quality
- Zero duplicate narration code
- Zero hardcoded ports
- Single source of truth for configuration
- 360+ LOC saved

### Functionality
- All 4 services with working UIs
- Hot reload in development
- Narration flows end-to-end
- Both dev and prod modes work

### Maintainability
- 40% reduction in maintenance burden
- Fix bugs in one place
- Easy to add new services
- Consistent patterns

---

## Resources

### Reference Documents
- `TEAM_350_QUICK_REFERENCE.md` - Quick patterns
- `TEAM_350_COMPLETE_IMPLEMENTATION_GUIDE.md` - Full guide
- `TEAM_350_ARCHITECTURE_RECOMMENDATIONS.md` - Architecture

### Phase Documents
- `TEAM_351_SHARED_PACKAGES_PHASE.md`
- `TEAM_352_QUEEN_MIGRATION_PHASE.md`
- `TEAM_353_HIVE_UI_PHASE.md`
- `TEAM_354_WORKER_UI_PHASE.md`
- `TEAM_355_FINAL_DOCUMENTATION_PHASE.md`

---

## Start Here

**First time?** Read this order:

1. `TEAM_350_QUICK_REFERENCE.md` - Get oriented (10 min)
2. Your phase document - Detailed instructions (30 min)
3. Execute your phase - Follow checklist
4. Create handoff - Summary for next team

**Questions?** Check troubleshooting sections in phase docs.

---

**Ready to start? Begin with TEAM-351!** 🚀
