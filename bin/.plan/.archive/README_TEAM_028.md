# TEAM-028 Start Here

**Welcome, TEAM-028!**

TEAM-027 has completed their work. Your mission: **QA everything and complete the MVP.**

---

## 🚨 START HERE 🚨

**Read these documents in order:**

### 1. Dev Rules (5 min)
```
/home/vince/Projects/llama-orch/.windsurf/rules/dev-bee-rules.md
```
**Why:** Understand the rules you must follow

### 2. QA Handoff (15 min)
```
/home/vince/Projects/llama-orch/bin/.plan/TEAM_028_HANDOFF_FINAL.md
```
**Why:** Learn what to test and how to be skeptical

### 3. MVP Spec (30 min)
```
/home/vince/Projects/llama-orch/bin/.specs/.gherkin/test-001-mvp.md
```
**Why:** Understand what the MVP should do

### 4. Implementation Handoff (15 min)
```
/home/vince/Projects/llama-orch/bin/.plan/TEAM_028_HANDOFF.md
```
**Why:** Get Phase 7-8 implementation templates

---

## Quick Summary

### What TEAM-027 Built

- ✅ rbee-hive daemon (HTTP server, background loops)
- ✅ rbee-keeper HTTP client
- ✅ worker-registry shared crate
- ✅ 8-phase MVP flow (Phases 1-6)
- ✅ Shared crates cleanup

### What's Incomplete

- ❌ Phase 7: Worker ready polling
- ❌ Phase 8: Inference execution
- ❌ End-to-end testing
- ❌ Integration with shared crates

### Your Mission

1. **QA TEAM-027's work** (be skeptical!)
2. **Implement Phase 7-8**
3. **Test end-to-end**
4. **Integrate shared crates**
5. **Fix bugs**

---

## Document Index

### Handoffs
- `TEAM_027_FINAL_HANDOFF.md` - Summary
- `TEAM_028_HANDOFF_FINAL.md` - **QA-focused (READ FIRST)**
- `TEAM_028_HANDOFF.md` - Implementation details

### Completion Reports
- `TEAM_027_COMPLETION_SUMMARY.md` - What was built
- `TEAM_027_QUICK_SUMMARY.md` - Quick reference
- `TEAM_027_ADDENDUM_SHARED_CRATES.md` - Shared crates work

### Shared Crates
- `../shared-crates/CRATE_USAGE_SUMMARY.md` - Analysis
- `../shared-crates/CLEANUP_COMPLETED.md` - Cleanup report

---

## Quick Start

### Verify Builds
```bash
cd /home/vince/Projects/llama-orch
cargo build --workspace
cargo test --workspace
```

### Start rbee-hive
```bash
cargo run --bin rbee-hive -- daemon
```

### Test Health Endpoint
```bash
curl http://localhost:8080/v1/health
```

### Try Inference (will fail at Phase 7)
```bash
cargo run --bin rbee -- infer \
  --node localhost \
  --model "test" \
  --prompt "test" \
  --max-tokens 5
```

---

## Key Files to Review

### rbee-hive
- `bin/rbee-hive/src/commands/daemon.rs` - Daemon entry point
- `bin/rbee-hive/src/monitor.rs` - Health monitoring
- `bin/rbee-hive/src/timeout.rs` - Idle timeout
- `bin/rbee-hive/src/http/workers.rs` - Worker endpoints

### rbee-keeper
- `bin/rbee-keeper/src/pool_client.rs` - HTTP client
- `bin/rbee-keeper/src/commands/infer.rs` - MVP flow (Phase 7-8 TODOs)

### Shared
- `bin/shared-crates/worker-registry/src/lib.rs` - Worker registry
- `bin/shared-crates/hive-core/` - Renamed from pool-core

---

## Remember

**TEAM-027 says: "Be skeptical. Test everything. Find bugs."**

They built quickly. Assume bugs exist. Your job is to find them and fix them.

Good luck! 🔍

---

**Created by:** TEAM-027  
**For:** TEAM-028  
**Date:** 2025-10-09T23:51:00+02:00
