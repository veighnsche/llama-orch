# Orchestratord Responsibility Audit - Executive Summary

**Date**: 2025-09-30  
**Auditor**: AI Assistant  
**Scope**: Verify orchestratord boundaries across entire codebase

---

## 🎯 Verdict: **CLEAN BOUNDARIES** ✅

No violations found. All crates respect their responsibilities.

---

## Quick Reference

### Orchestratord IS Responsible For:
- ✅ HTTP API surface (control + data plane)
- ✅ Admission validation & enqueue coordination
- ✅ SSE streaming orchestration
- ✅ Placement decisions (query pool-managerd, select target)
- ✅ Handoff autobind watcher (react to provisioner outputs)
- ✅ Metrics aggregation
- ✅ Correlation ID middleware
- ✅ Session management HTTP endpoints
- ✅ Artifact registry HTTP endpoints
- ✅ Catalog HTTP endpoints

### Orchestratord is NOT Responsible For:
- ❌ Engine provisioning → `engine-provisioner`
- ❌ Model fetching → `model-provisioner`
- ❌ Queue implementation → `orchestrator-core`
- ❌ Catalog storage → `catalog-core`
- ❌ Pool supervision → `pool-managerd`
- ❌ Adapter implementation → `worker-adapters/*`
- ❌ Token generation → engines (llama.cpp, vLLM, etc.)

---

## Key Design Patterns (Validated)

### 1. **Handoff Pattern** ✅
```
engine-provisioner → writes .runtime/engines/*.json
orchestratord → watches, binds adapters, updates registry
```
**Why**: Avoids circular dependencies, clean separation

### 2. **Embedded Libraries (Home Profile)** ✅
```
orchestratord embeds:
  - pool-managerd::Registry
  - catalog-core::FsCatalog
  - orchestrator-core::InMemoryQueue
  - adapter-host::AdapterHost
```
**Why**: Single binary, lightweight, optimized for home lab

### 3. **HTTP API Layer** ✅
```
Libraries provide logic/storage:
  - catalog-core (storage)
  - pool-managerd (registry)
  
orchestratord provides HTTP surface:
  - POST /v2/catalog/models
  - GET /v2/pools/:id/health
```
**Why**: Standard layering, context injection (auth, correlation ID, metrics)

---

## Potential Concerns (All Resolved)

| Concern | Resolution | Status |
|---------|-----------|--------|
| Handoff watcher in orchestratord? | Coordination is orchestratord's job | ✅ Correct |
| Pool registry embedded? | Home profile pattern, documented | ✅ Correct |
| Catalog HTTP in orchestratord? | Standard API layer over storage lib | ✅ Correct |

---

## Recommendations

1. ✅ **Keep current design** - No changes needed
2. 📝 **Add architecture diagram** to README
3. 📝 **Document handoff pattern** as design pattern
4. 📝 **Clarify home vs cloud profile** in docs

---

## Files Reviewed

- `.specs/00_llama-orch.md` - Overall architecture
- `.specs/20-orchestratord.md` - Orchestratord spec
- `.specs/30-pool-managerd.md` - Pool manager spec
- `.specs/50-engine-provisioner.md` - Provisioner spec
- `bin/orchestratord/src/**/*.rs` - All orchestratord code
- `libs/pool-managerd/src/**/*.rs` - Pool manager code
- `libs/provisioners/engine-provisioner/src/**/*.rs` - Provisioner code
- `libs/catalog-core/src/**/*.rs` - Catalog code
- `libs/orchestrator-core/src/**/*.rs` - Queue code

---

## Conclusion

**orchestratord is correctly scoped.** No overreach. No gaps. Design follows spec. 🎯

See `RESPONSIBILITY_AUDIT.md` for detailed analysis.
