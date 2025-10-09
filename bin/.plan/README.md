# TEAM-022 Work Breakdown: Infrastructure for Multi-Model Testing

**Created by:** TEAM-022  
**Date:** 2025-10-09  
**Status:** Active

---

## Overview

This directory contains the checkpoint-based work breakdown for TEAM-022's mission to build CLI infrastructure for multi-model testing.

**Context:** TEAM-021 fixed the Metal broadcasting bug. llorch-candled works correctly. Now we need infrastructure to test multiple models across pools.

---

## Checkpoint Structure

Each checkpoint represents **1 week of equal work** and ends with a **testable milestone**.

| Checkpoint | Focus | Duration | Dependencies |
|------------|-------|----------|--------------|
| **CP1** | Foundation (shared crates + CLI skeletons) | Week 1 | None |
| **CP2** | Model Catalog System | Week 2 | CP1 |
| **CP3** | Automation (downloads + spawning) | Week 3 | CP2 |
| **CP4** | Multi-Model Testing | Week 4 | CP3 |

---

## Files

- `00_TEAM_022_MASTER_PLAN.md` - Executive summary and strategy
- `01_CP1_FOUNDATION.md` - Shared crates + CLI skeletons
- `02_CP2_MODEL_CATALOG.md` - Catalog system implementation
- `03_CP3_AUTOMATION.md` - Model downloads + worker spawning
- `04_CP4_MULTI_MODEL.md` - Multi-model testing (THE GOAL)

---

## Checkpoint Gates

**CRITICAL:** After each checkpoint, we **STOP** and verify all functionality before proceeding.

### CP1 Gate
- [ ] pool-core crate compiles
- [ ] pool-ctl binary works
- [ ] llorch-ctl binary works
- [ ] SSH connectivity verified

### CP2 Gate
- [ ] Catalog system works
- [ ] Models can be registered
- [ ] Catalogs created on all pools
- [ ] Remote catalog access works

### CP3 Gate
- [ ] Model downloads work
- [ ] Worker spawning works
- [ ] Qwen downloaded on all pools
- [ ] Qwen tested on Metal and CUDA

### CP4 Gate (FINAL)
- [ ] All 4 models downloaded
- [ ] All models tested on all backends
- [ ] 100% test pass rate
- [ ] Documentation complete

---

## Architecture Alignment

**From:** `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`

**Control Plane:** SSH (operator → pools)  
**Data Plane:** HTTP (orchestrator → workers)

**Binaries:**
- `pool-ctl` - Local pool management (command: `llorch-pool`)
- `llorch-ctl` - Remote pool control via SSH (command: `llorch`)
- `llorch-candled` - Worker daemon (HTTP server, already exists)

**Note:** `worker-orcd` is **DEPRECATED**. We use `llorch-candled` now.

---

## Success Criteria

**Mission Complete When:**

```
Model Support Matrix

| Model | Metal (Mac) | CUDA (Workstation) | Notes |
|-------|-------------|---------------------|-------|
| TinyLlama 1.1B | ✅ | ✅ | Fully tested |
| Qwen 0.5B | ✅ | ✅ | Fully tested |
| Phi-3 Mini | ✅ | ✅ | Fully tested |
| Mistral 7B | ✅ | ✅ | Fully tested |
```

**Total:** 8 tests (4 models × 2 backends), 100% pass rate

---

## Timeline

**Week 1:** CP1 - Foundation  
**Week 2:** CP2 - Model Catalog  
**Week 3:** CP3 - Automation  
**Week 4:** CP4 - Multi-Model Testing

**Total:** 4 weeks

---

## How to Use This Plan

1. **Start with Master Plan:** Read `00_TEAM_022_MASTER_PLAN.md`
2. **Execute Checkpoints Sequentially:** CP1 → CP2 → CP3 → CP4
3. **Stop at Each Gate:** Verify all criteria before proceeding
4. **Document Progress:** Update checkpoint status as you go
5. **Celebrate at CP4:** Mission accomplished! 🎉

---

## References

**Specs:**
- `/bin/.specs/00_llama-orch.md` - System architecture
- `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Control/data plane
- `/bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md` - Binary structure

**Handoff:**
- `/bin/llorch-candled/.specs/TEAM_022_HANDOFF.md` - Detailed task breakdown from TEAM-021

**Rules:**
- `/.windsurf/rules/candled-rules.md` - Team coding standards

---

## Team Signatures

All code created during TEAM-022 work must include:

```rust
// Created by: TEAM-022
```

Or for modifications:

```rust
// TEAM-022: <description of change>
```

---

**Next Step:** Read `00_TEAM_022_MASTER_PLAN.md` to begin.
