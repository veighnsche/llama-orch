# TEAM-022 Master Plan: Infrastructure for Multi-Model Testing

**Created by:** TEAM-022  
**Date:** 2025-10-09  
**Status:** Active  
**Goal:** Build CLI infrastructure to enable multi-model testing across pools

---

## Executive Summary

**Context:** TEAM-021 fixed the Metal broadcasting bug (cache pollution). llm-worker-rbee works correctly on all backends.

**Blocker:** No infrastructure to test multiple models across multiple pools.

**Mission:** Build `rbee-hive` and `rbee-keeper` binaries to enable automated multi-model testing.

---

## Architecture Alignment

**From:** `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`

**Control Plane:** SSH (operator → pools)  
**Data Plane:** HTTP (orchestrator → workers)

**Binaries to Build:**
1. `rbee-hive` (local pool management CLI)
2. `rbee-keeper` (remote pool control via SSH)

**Note:** `worker-orcd` is **DEPRECATED**. We now use `llm-worker-rbee` for all worker operations.

---

## Checkpoint Strategy

**Equal Work Distribution:** Each checkpoint contains ~1 week of work and ends with a testable milestone.

**Checkpoints:**
1. **CP1: Foundation** - Shared crates + basic CLI skeletons
2. **CP2: Model Catalog** - Catalog system + registration
3. **CP3: Automation** - Model downloads + worker spawning
4. **CP4: Multi-Model Testing** - Download all models + test script

**Test Points:** After each checkpoint, we STOP and verify all functionality before proceeding.

---

## Checkpoint Files

- `01_CP1_FOUNDATION.md` - Shared crates + CLI skeletons
- `02_CP2_MODEL_CATALOG.md` - Catalog system
- `03_CP3_AUTOMATION.md` - Downloads + spawning
- `04_CP4_MULTI_MODEL.md` - Multi-model testing

---

## Success Criteria

**CP1 Complete:**
- [ ] `pool-core` crate compiles
- [ ] `rbee-hive` binary compiles and runs
- [ ] `rbee-keeper` binary compiles and runs
- [ ] Basic commands work (help, version)

**CP2 Complete:**
- [ ] Catalog format defined (JSON schema)
- [ ] Catalog can be created/loaded/saved
- [ ] Models can be registered/unregistered
- [ ] `rbee-hive models catalog` shows models

**CP3 Complete:**
- [ ] `rbee-hive models download <model>` works
- [ ] `rbee-hive worker spawn <backend> --model <model>` works
- [ ] Qwen downloaded on all pools
- [ ] Qwen tested on Metal and CUDA

**CP4 Complete:**
- [ ] All 4 models downloaded (TinyLlama, Qwen, Phi, Mistral)
- [ ] Test script created
- [ ] All models tested on all backends
- [ ] Results documented in MODEL_SUPPORT.md

---

## Timeline

**Week 1:** CP1 - Foundation  
**Week 2:** CP2 - Model Catalog  
**Week 3:** CP3 - Automation  
**Week 4:** CP4 - Multi-Model Testing

---

## References

**Specs:**
- `/bin/.specs/00_llama-orch.md` - System architecture
- `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Control/data plane split
- `/bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md` - Binary structure

**Handoff:**
- `/bin/llm-worker-rbee/.specs/TEAM_022_HANDOFF.md` - Detailed task breakdown

**Rules:**
- `/.windsurf/rules/candled-rules.md` - Team coding standards

---

**Next:** Read `01_CP1_FOUNDATION.md` to start implementation.
