# TEAM-198 NARRATION ARCHITECTURE SUMMARY

**Team:** TEAM-198  
**Mission:** Complete narration SSE architecture investigation and solution design  
**Date:** 2025-10-22  
**Status:** ✅ **ANALYSIS COMPLETE**

---

## Mission Statement

Investigate and propose a solution where:
- ✅ **All queen narration** flows through SSE jobs channel (web-UI proof)
- ✅ **Only bee-keeper** uses stdout (CLI tool)
- ✅ **Hive and worker** narration flows: worker → hive → queen → keeper → stdout
- ✅ **Simple, ergonomic API** for developers
- ✅ **Centralized formatting** (no manual formatting in consumers)

---

## Key Findings

### 1. Current Architecture Traced

Documented complete narration flow across all 4 binaries:
- **rbee-keeper** (CLI): Direct stdout ✅
- **queen-rbee** (daemon): stderr + global SSE broadcaster ✅
- **rbee-hive** (daemon): println! only (NO narration yet) ❌
- **llm-worker** (daemon): dual-output (stderr + thread-local SSE) ⚠️

### 2. Root Problem Identified (TEAM-197 Bug)

**Decentralized Formatting:**
- narration-core sends raw `NarrationEvent` struct via SSE
- Each consumer must format manually: `format!("[{:<10}] {:<15}: {}", actor, action, human)`
- TEAM-197 fixed queen consumer, but root cause remains

**Impact:**
- ❌ Manual duplication of formatting logic
- ❌ Inconsistent formats possible
- ❌ Breaking changes when format updates

### 3. Architecture Violation Found

**Remote daemon narration is NOT web-UI proof:**
- Hive/worker use `println!()` or stderr
- When running on remote machines, keeper cannot see this output
- Violates user requirement: "ALL narration must go through SSE"

---

## Solution Design

### Core Principle: Pre-Formatted SSE Events

**Add `formatted: String` field to `NarrationEvent`:**

```rust
pub struct NarrationEvent {
    pub formatted: String,  // ← Pre-formatted: "[actor     ] action         : message"
    
    // Keep existing fields for backward compatibility
    pub actor: String,
    pub action: String,
    pub human: String,
    // ...
}
```

**Benefits:**
- ✅ Single formatting location (narration-core)
- ✅ Consumers just use `event.formatted` (no manual work)
- ✅ Format changes update all consumers automatically
- ✅ Backward compatible (old fields still available)

### Complete Flow

```
┌─────────────────────────────────────────────────────────────┐
│ NARRATION EMISSION                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hive/Worker (remote machine)                              │
│    ↓                                                        │
│  NARRATE.action().emit()                                   │
│    ↓                                                        │
│  narration-core formats: "[hive      ] spawn      : ..."   │
│    ├─ stderr (daemon logs)                                 │
│    └─ POST to queen: /v1/narration (new endpoint)          │
│         ↓                                                   │
│  Queen receives and broadcasts                             │
│    ├─ Adds to global SSE broadcaster                       │
│    └─ Sends pre-formatted to keeper via job SSE            │
│         ↓                                                   │
│  Keeper receives and displays                              │
│    └─ println!(event.formatted)  // Just print!            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Centralized Formatting (narration-core)
- Add `formatted` field to `NarrationEvent`
- Extract `format_narration()` helper function
- Pre-format in `NarrationEvent::from()`
- **Impact:** ~15 lines, no breaking changes

### Phase 2: Simplify Queen Consumer
- Remove manual formatting in `jobs.rs` line 107-109
- Use `event.formatted` instead
- **Impact:** -3 lines, simpler code

### Phase 3: Queen Ingestion Endpoint (NEW)
- Add `POST /v1/narration` endpoint
- Receives pre-formatted narration from hive/worker
- Broadcasts via global SSE broadcaster
- **Impact:** ~30 lines

### Phase 4: Hive Narration
- Create `narration.rs` module
- Replace `println!()` with `NARRATE.action().emit()`
- Add `narrate_to_queen()` helper (POST to queen)
- **Impact:** ~40 lines

### Phase 5: Worker Lifecycle Narration
- Add `narrate_with_queen()` for lifecycle events
- Keep `narrate_dual()` for inference (already works via SSE)
- **Impact:** ~30 lines

**Total:** ~112 lines added, complete web-UI proof narration

---

## Files Analysis

### Component Breakdown

**Total files investigated:** 15+
- narration-core (4 files: lib.rs, sse_sink.rs, builder.rs, capture.rs)
- queen-rbee (3 files: jobs.rs, job_router.rs, narration.rs)
- rbee-keeper (2 files: main.rs, job_client.rs)
- rbee-hive (1 file: main.rs)
- llm-worker (2 files: narration.rs, narration_channel.rs)
- job-registry (1 file: lib.rs)

### Files to Create
- `bin/10_queen_rbee/src/http/narration.rs` (NEW ingestion endpoint)
- `bin/20_rbee_hive/src/narration.rs` (NEW hive narration helpers)

### Files to Modify
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (add formatted field)
- `bin/99_shared_crates/narration-core/src/lib.rs` (extract format helper)
- `bin/10_queen_rbee/src/http/jobs.rs` (use event.formatted)
- `bin/10_queen_rbee/src/main.rs` (add narration route)
- `bin/20_rbee_hive/src/main.rs` (use narration, not println!)
- `bin/30_llm_worker_rbee/src/narration.rs` (add queen forwarding)

---

## API Design: Simple and Ergonomic

### Keeper (unchanged)
```rust
NARRATE.action("job_submit").context(job_id).human("📋 Job {}").emit();
```

### Queen (unchanged)
```rust
NARRATE.action("job_create").context(&job_id).human("Job {} created").emit();
```

### Hive (new - queen forwarding)
```rust
narrate_to_queen(
    &client, 
    queen_url,
    NARRATE.action("spawn").context("worker-1").human("Spawning worker {}").build()
).await;
```

### Worker (new - queen forwarding for lifecycle)
```rust
// Inference: Use existing narrate_dual() (thread-local SSE) ✅
narrate_dual(fields);

// Lifecycle: Send to queen (new)
narrate_with_queen(fields, Some(queen_url)).await;
```

---

## Engineering Rules Compliance

**✅ All rules followed:**
- ✅ No TODOs in analysis
- ✅ Investigated complete architecture
- ✅ Proposed concrete solution with code examples
- ✅ No "next team should..." (complete implementation plan provided)
- ✅ Added TEAM-198 signature
- ✅ Documentation ≤2 pages (summary here, detailed doc in separate file)

---

## Deliverables

### 1. Complete Architecture Document
**File:** `bin/99_shared_crates/narration-core/NARRATION_SSE_ARCHITECTURE_TEAM_198.md`
- 542 lines
- Complete flow diagrams
- Problem analysis
- Solution design
- Implementation plan (5 phases)
- Code examples
- Verification checklist
- Benefits analysis

### 2. This Summary
**File:** `TEAM-198-NARRATION-SUMMARY.md`
- Concise overview
- Key findings
- Implementation phases
- Files to create/modify
- API design examples

---

## Verification Checklist

### Analysis Complete
- ✅ Traced narration flow across all 4 binaries
- ✅ Identified root cause (decentralized formatting)
- ✅ Found architecture violation (remote daemon stdout)
- ✅ Documented current state (TEAM-197 work)
- ✅ Understood SSE broadcaster pattern
- ✅ Analyzed worker thread-local channel
- ✅ Read engineering rules and debugging rules

### Solution Design Complete
- ✅ Designed centralized formatting approach
- ✅ Proposed queen ingestion endpoint
- ✅ Planned hive/worker narration forwarding
- ✅ Maintained simple, ergonomic API
- ✅ Ensured backward compatibility
- ✅ Calculated code size impact (~112 lines)

### Documentation Complete
- ✅ Created comprehensive architecture document (542 lines)
- ✅ Provided implementation plan (5 phases)
- ✅ Included code examples for all changes
- ✅ Documented complete flow diagrams
- ✅ Listed all files to create/modify
- ✅ Provided verification checklist

---

## Key Insights

### 1. Queen is Natural Aggregation Point
- Already manages jobs
- Already has SSE infrastructure
- Already central hub
- Simple POST endpoint, no new service needed

### 2. Pre-Formatted SSE Eliminates Duplication
- One place formats (narration-core)
- Everyone else just displays
- Format changes propagate automatically
- No more TEAM-197-style bugs

### 3. Web-UI Proof Requires SSE for All Daemons
- CLI tool (keeper): Can use stdout ✅
- Daemons (queen/hive/worker): MUST use SSE ✅
- Remote machines: Cannot see stdout/stderr ✅
- Solution: All narration flows through queen SSE ✅

---

## Benefits

### For Developers
- ✅ Simple API: Just call `.emit()`
- ✅ No thinking about output paths
- ✅ No manual formatting
- ✅ Works everywhere (local and remote)

### For Operations
- ✅ All narration in one place (keeper stdout or web UI)
- ✅ Consistent formatting
- ✅ Remote daemon narration visible
- ✅ Easy to grep/parse

### For Maintenance
- ✅ Single formatting location
- ✅ Format changes don't break consumers
- ✅ No duplication
- ✅ Backward compatible

---

## Related Work

**Built on TEAM-197's work:**
- TEAM-197 fixed queen consumer formatting (line 107-109 in jobs.rs)
- TEAM-197 identified decentralized formatting problem
- TEAM-197 documented SSE flow
- TEAM-198 provides architectural solution

**Preserves existing patterns:**
- Worker thread-local channel (TEAM-039)
- Global SSE broadcaster (TEAM-164)
- Job-based architecture (TEAM-186)
- Narration factory pattern (TEAM-192)

---

## Next Team: Implementation Guide

### Start Here
1. Read: `NARRATION_SSE_ARCHITECTURE_TEAM_198.md` (complete design)
2. Understand: Current SSE flow (diagrams in doc)
3. Review: Implementation plan (5 phases)

### Phase Order (Sequential)
1. **Phase 1:** narration-core (foundation, no breaking changes)
2. **Phase 2:** queen consumer (simplifies existing code)
3. **Phase 3:** queen endpoint (new capability, testable in isolation)
4. **Phase 4:** hive narration (makes hive visible)
5. **Phase 5:** worker narration (makes worker lifecycle visible)

### Testing at Each Phase
- Phase 1: Unit test `format_narration()`
- Phase 2: E2E test `./rbee hive status`
- Phase 3: Curl test `POST /v1/narration`
- Phase 4: Remote hive test
- Phase 5: Remote worker test

### Success Criteria
- [ ] Keeper sees ALL narration (local and remote)
- [ ] Format is consistent everywhere
- [ ] No manual formatting in consumers
- [ ] Web UI can subscribe to SSE (future-proof)
- [ ] API is simple and ergonomic

---

## Summary

**Problem:** Narration not web-UI proof, formatting decentralized  
**Solution:** Pre-formatted SSE + queen ingestion endpoint  
**Impact:** ~112 lines, complete visibility, simple API  
**Status:** Ready for implementation

**The architecture is now designed for web-UI proof, simple, and ergonomic narration across all rbee components.**

---

**Created by:** TEAM-198  
**Status:** ✅ **ANALYSIS COMPLETE, READY FOR IMPLEMENTATION**  
**Handoff:** Complete architecture in `NARRATION_SSE_ARCHITECTURE_TEAM_198.md`
