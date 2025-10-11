# Test-001.md Update Summary
# Updated by: TEAM-077
# Date: 2025-10-11
# Status: ✅ COMPLETE

## What Was Updated

The test-001.md reference document has been comprehensively updated with all latest understanding:

### 1. Component Naming (TEAM-077) ✅
**Added section with correct naming:**
- **queen-rbee** - Orchestrator daemon (not "orchestratord")
- **rbee-hive** - Pool manager daemon (not "pool-managerd")
- **worker-rbee** - Worker daemon (llm-worker-rbee, sd-worker-rbee, etc.)
- **rbee-keeper** - CLI/UI tool

### 2. GPU FAIL FAST Policy (TEAM-075) ✅
**Added comprehensive policy section:**
- ❌ NO automatic backend fallback (GPU → CPU)
- ❌ NO graceful degradation
- ❌ NO CPU fallback on GPU failure
- ✅ FAIL FAST with exit code 1
- ✅ Clear error message with actionable suggestions
- ✅ User must explicitly choose backend

**Error Codes:**
- `CUDA_DEVICE_FAILED`
- `GPU_VRAM_EXHAUSTED`
- `GPU_NOT_AVAILABLE`

**Exit Code:** 1 (FAIL FAST)

### 3. Updated Error Handling Scenarios ✅
**Modified error scenarios to reflect FAIL FAST policy:**

**EH-005a: VRAM Exhausted (GPU FAIL FAST)**
- Response: FAIL FAST with exit code 1 (NO CPU fallback)
- Policy: User must explicitly choose CPU backend, NO automatic fallback

**EH-009a: Backend Not Available (GPU FAIL FAST)**
- Response: FAIL FAST with exit code 1 (NO automatic fallback)
- Policy: User must explicitly choose available backend, NO automatic selection

**EH-009b: CUDA Not Installed (GPU FAIL FAST)**
- Response: FAIL FAST with exit code 1 (NO CPU fallback)
- Suggestion: "Install CUDA OR use CPU explicitly: --backend cpu"
- Policy: User must fix CUDA installation or explicitly choose CPU

### 4. Milestone Alignment (TEAM-077) ✅
**Added comprehensive milestone section:**

**M0 (v0.1.0) - Worker Haiku Test**
- Goal: Worker binary runs standalone
- Components: worker-rbee only
- Scenarios: Worker lifecycle, inference execution

**M1 (v0.2.0) - Pool Manager Lifecycle**
- Goal: rbee-hive can start/stop workers, hot-load models
- Components: rbee-hive + worker-rbee
- Scenarios: All scenarios in this document
- New Components Needed:
  - Worker binaries catalog (track which workers installed)
  - SSH preflight checks (validate SSH before spawning)
  - rbee-hive preflight checks (validate readiness)

**M2 (v0.3.0) - Orchestrator Scheduling**
- Goal: queen-rbee with Rhai scheduler
- Components: queen-rbee + Rhai scheduler
- Status: Documented but deferred

**M3 (v0.4.0) - Security & Platform**
- Goal: auth, audit logging, multi-tenancy
- Components: auth-min, audit-logging, secrets-management
- Status: Documented but deferred

### 5. Feature File Mapping (TEAM-077) ✅
**Added complete mapping to feature files:**

**M0-M1 Feature Files (13 files):**
- 010-ssh-registry-management.feature
- 020-model-catalog.feature
- 025-worker-binaries-catalog.feature (NEW!)
- 040-rbee-hive-worker-registry.feature
- 050-ssh-preflight-checks.feature (NEW!)
- 060-rbee-hive-preflight-checks.feature (NEW!)
- 070-worker-preflight-checks.feature
- 080-worker-rbee-lifecycle.feature
- 090-rbee-hive-lifecycle.feature
- 110-inference-execution.feature
- 120-input-validation.feature
- 130-cli-commands.feature
- 140-end-to-end-flows.feature

**M2+ Feature Files (9 files - documented but deferred):**
- 030-queen-rbee-worker-registry.feature (M2)
- 100-queen-rbee-lifecycle.feature (M2)
- 150-authentication.feature (M3)
- 160-audit-logging.feature (M3)
- 170-input-validation.feature (M3)
- 180-secrets-management.feature (M3)
- 190-deadline-propagation.feature (M3)
- 200-rhai-scheduler.feature (M2)
- 210-queue-management.feature (M2)

### 6. Updated Revision History ✅
**Added TEAM-075 and TEAM-077 entries:**
- **TEAM-075** (2025-10-10): Added GPU FAIL FAST policy, removed all fallback chains, enforced clear error modes
- **TEAM-077** (2025-10-11): Updated naming conventions (rbee-hive, worker-rbee), added milestone alignment, mapped to feature files

### 7. Updated Status Line ✅
**New status:**
```
Status: ✅ CORRECTED + ENHANCED + ERROR HANDLING + GPU FAIL FAST + MILESTONE ALIGNED
```

## Key Principles Preserved

### 1. Reference Document Integrity ✅
- Each section contains ALL information needed by engineers
- No information removed, only enhanced
- All error scenarios preserved with policy clarifications
- Complete flow documentation maintained

### 2. GPU FAIL FAST Policy Enforcement ✅
- Found and incorporated TEAM-075's critical policy
- Applied to all relevant error scenarios
- Clear "NO FALLBACK" messaging
- Actionable suggestions for users

### 3. Naming Consistency ✅
- Updated all component references to use correct names
- Added naming section for clarity
- Maintained consistency throughout document

### 4. Milestone Awareness ✅
- Document now clearly states it's for M0-M1
- New M1 components identified
- M2+ features documented but marked as deferred
- Engineers know what's in scope vs future

## What Was NOT Changed

### Preserved Content ✅
- All error handling scenarios (EH-001 through EH-018)
- All narration flow examples
- All code examples and schemas
- All topology and test objectives
- All phase-by-phase flow documentation
- All timeout values and retry strategies
- All cancellation handling (Gap-G12)

### Why Preserved
Each section is a reference for engineers implementing BDD tests. Removing or condensing would lose critical implementation details.

## Document Purpose

**test-001.md is a REFERENCE DOCUMENT for:**
1. BDD test implementation
2. Error handling scenarios
3. Narration flow patterns
4. Component interaction flows
5. Timeout and retry strategies
6. GPU FAIL FAST policy enforcement

**Engineers use this to:**
- Implement step definitions
- Understand error scenarios
- Follow narration patterns
- Apply correct policies (FAIL FAST)
- Map scenarios to feature files

## Verification Checklist

- ✅ All TEAM-075 FAIL FAST rules incorporated
- ✅ All component names updated to TEAM-077 conventions
- ✅ Milestone alignment added
- ✅ Feature file mapping added
- ✅ Revision history updated
- ✅ Status line updated
- ✅ No information removed
- ✅ All sections remain complete
- ✅ Document remains engineer-friendly reference

## Summary

test-001.md has been carefully updated to reflect:
- ✅ Correct component naming (rbee-hive, worker-rbee, queen-rbee)
- ✅ GPU FAIL FAST policy (NO automatic fallback)
- ✅ Milestone alignment (M0-M1 scope, M2+ deferred)
- ✅ Feature file mapping (13 M0-M1 files, 9 M2+ files)
- ✅ Complete revision history
- ✅ All original content preserved

**Status:** ✅ REFERENCE DOCUMENT UPDATED AND ENHANCED

---

**TEAM-077 says:** test-001.md updated! GPU FAIL FAST policy incorporated! Component naming corrected! Milestone alignment added! Feature file mapping complete! All information preserved! Ready for BDD implementation! 🐝
