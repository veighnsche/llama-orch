# 📦 Cargo.toml Inventory

**Total Count**: 49 files  
**Date**: 2025-10-04

---

## 🎯 KEY FILE

**`/home/vince/Projects/llama-orch/Cargo.toml`** - **WORKSPACE ROOT**
- This is the ONLY file that needs the `[workspace.dependencies]` update
- All other Cargo.toml files inherit from this workspace

---

## 📋 ALL CARGO.TOML FILES

### Root (1)
1. `/home/vince/Projects/llama-orch/Cargo.toml` ⭐ **WORKSPACE ROOT**

### Main Services (6)
2. `/home/vince/Projects/llama-orch/bin/rbees-orcd/Cargo.toml`
3. `/home/vince/Projects/llama-orch/bin/rbees-orcd/bdd/Cargo.toml`
4. `/home/vince/Projects/llama-orch/bin/pool-managerd/Cargo.toml`
5. `/home/vince/Projects/llama-orch/bin/pool-managerd/bdd/Cargo.toml`
6. `/home/vince/Projects/llama-orch/bin/worker-orcd/Cargo.toml`
7. `/home/vince/Projects/llama-orch/bin/worker-orcd/bdd/Cargo.toml`

### Orchestratord Crates (9)
8. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/agentic-api/Cargo.toml`
9. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/backpressure/Cargo.toml`
10. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/job-timeout/Cargo.toml`
11. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/orchestrator-core/Cargo.toml`
12. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/orchestrator-core/bdd/Cargo.toml`
13. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/platform-api/Cargo.toml`
14. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/pool-registry/Cargo.toml`
15. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/scheduling/Cargo.toml`
16. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/streaming/Cargo.toml`
17. `/home/vince/Projects/llama-orch/bin/rbees-orcd-crates/task-cancellation/Cargo.toml`

### Pool-Managerd Crates (10)
18. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/capability-matcher/Cargo.toml`
19. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/control-api/Cargo.toml`
20. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/error-ops/Cargo.toml`
21. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/gpu-inventory/Cargo.toml`
22. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/model-cache/Cargo.toml`
23. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/model-catalog/Cargo.toml`
24. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/model-catalog/bdd/Cargo.toml`
25. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/model-provisioner/Cargo.toml`
26. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/model-provisioner/bdd/Cargo.toml`
27. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/pool-registration-client/Cargo.toml`
28. `/home/vince/Projects/llama-orch/bin/pool-managerd-crates/worker-lifecycle/Cargo.toml`

### Shared Crates (13)
29. `/home/vince/Projects/llama-orch/bin/shared-crates/audit-logging/Cargo.toml`
30. `/home/vince/Projects/llama-orch/bin/shared-crates/audit-logging/bdd/Cargo.toml`
31. `/home/vince/Projects/llama-orch/bin/shared-crates/auth-min/Cargo.toml`
32. `/home/vince/Projects/llama-orch/bin/shared-crates/deadline-propagation/Cargo.toml`
33. `/home/vince/Projects/llama-orch/bin/shared-crates/gpu-info/Cargo.toml`
34. `/home/vince/Projects/llama-orch/bin/shared-crates/input-validation/Cargo.toml`
35. `/home/vince/Projects/llama-orch/bin/shared-crates/input-validation/bdd/Cargo.toml`
36. `/home/vince/Projects/llama-orch/bin/shared-crates/narration-core/Cargo.toml`
37. `/home/vince/Projects/llama-orch/bin/shared-crates/narration-core/bdd/Cargo.toml`
38. `/home/vince/Projects/llama-orch/bin/shared-crates/narration-macros/Cargo.toml`
39. `/home/vince/Projects/llama-orch/bin/shared-crates/pool-registry-types/Cargo.toml`
40. `/home/vince/Projects/llama-orch/bin/shared-crates/secrets-management/Cargo.toml`
41. `/home/vince/Projects/llama-orch/bin/shared-crates/secrets-management/bdd/Cargo.toml`

### Consumer Libraries (2)
42. `/home/vince/Projects/llama-orch/consumers/llama-orch-sdk/Cargo.toml`
43. `/home/vince/Projects/llama-orch/consumers/llama-orch-utils/Cargo.toml`

### Contracts (2)
44. `/home/vince/Projects/llama-orch/contracts/api-types/Cargo.toml`
45. `/home/vince/Projects/llama-orch/contracts/config-schema/Cargo.toml`

### Tools (4)
46. `/home/vince/Projects/llama-orch/tools/openapi-client/Cargo.toml`
47. `/home/vince/Projects/llama-orch/tools/readme-index/Cargo.toml`
48. `/home/vince/Projects/llama-orch/tools/spec-extract/Cargo.toml`
49. `/home/vince/Projects/llama-orch/xtask/Cargo.toml`

---

## 🎯 ACTION REQUIRED

**ONLY UPDATE THIS ONE FILE:**
```
/home/vince/Projects/llama-orch/Cargo.toml
```

All 48 other Cargo.toml files will automatically inherit the pinned versions from the workspace root.

---

## ✅ HOW WORKSPACE DEPENDENCIES WORK

When you update the root `Cargo.toml` with:
```toml
[workspace.dependencies]
tokio = { version = "1.47", features = ["full"] }
```

All member crates can use it like:
```toml
[dependencies]
tokio = { workspace = true }
```

This means:
- ✅ Update 1 file → affects all 49 crates
- ✅ Consistent versions across entire monorepo
- ✅ No need to touch individual Cargo.toml files

---

## 🚀 NEXT STEPS

1. Open `/home/vince/Projects/llama-orch/Cargo.toml`
2. Replace the `[workspace.dependencies]` section with the pinned versions from `VERSION_PINNING_ACTION.md`
3. Run `cargo update --workspace`
4. Run `cargo build --workspace`
5. Done! All 49 crates now use pinned versions

---

**CREATED BY**: Cascade  
**Date**: 2025-10-04
