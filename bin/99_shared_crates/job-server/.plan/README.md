# Job Server Planning Documents

**Date:** October 26, 2025  
**Team:** TEAM-304

---

## Documents in This Directory

### 1. JOB_LIFECYCLE_ROBUSTIFICATION.md

**Purpose:** Analysis of whether we should reorganize job-server into a modular `job-lifecycle/` structure (like `daemon-lifecycle`)

**Key Findings:**
- ✅ **Current structure is GOOD** - No reorganization needed
- ✅ **File size is manageable** - 448 LOC in single file
- ✅ **Well-tested** - 42 tests across 4 test files
- ⚠️ **Enhancements needed** - Timeout, cancellation, retry

**Verdict:** Keep current structure, add functionality enhancements

**Recommended Enhancements:**
1. **Phase 1 (Critical):** Job timeout + cancellation
2. **Phase 2 (Important):** Retry logic + priority queue
3. **Phase 3 (Nice to have):** Metadata, history, persistence

---

### 2. EXTERNAL_CRATE_EVALUATION.md

**Purpose:** Research external Rust crates that could replace or complement our job-server and narration-core implementations

**Key Findings:**
- ✅ **Our custom solution is appropriate** - Well-suited for dual-call + SSE pattern
- ✅ **Adopt tokio-util** - For cancellation and timeout
- 🔶 **Consider tracing** - For backend observability (not urgent)
- ❌ **Avoid job queue libraries** - Wrong architectural model (apalis, fang, effectum)

**Evaluated Categories:**
1. Job queue libraries (apalis, fang, effectum)
2. SSE/streaming (axum-sse - already using)
3. Observability (tracing, opentelemetry)
4. State machines (smlang)
5. Async utilities (tokio-util - recommended)

**Verdict:** Keep our custom implementation, adopt tokio-util for specific features

---

## Quick Reference

### Current Architecture

```
job-server/
├── src/
│   └── lib.rs                    # 448 LOC
│       ├── JobRegistry           # In-memory state management
│       ├── JobState              # State machine (4 states)
│       └── execute_and_stream()  # Deferred execution + SSE
├── tests/
│   ├── done_signal_tests.rs      # 7 tests (TEAM-304)
│   ├── concurrent_access_tests.rs # 11 tests
│   ├── resource_cleanup_tests.rs  # 14 tests
│   └── job_registry_edge_cases_tests.rs # 24 tests
└── Cargo.toml
```

### Why Our Custom Solution?

**Our requirements are unique:**
1. **Dual-call pattern** - POST creates job, GET streams results
2. **SSE streaming** - Real-time feedback to clients
3. **Ephemeral jobs** - No persistence needed
4. **Single-node** - No distributed coordination
5. **Tight integration** - With narration-core for observability

**External job queues (apalis, fang, effectum) are designed for:**
1. Worker pools (not dual-call)
2. Database persistence (not ephemeral)
3. Traditional background jobs (not SSE streaming)
4. Distributed systems (not single-node)

**Conclusion:** Our architecture is fundamentally different. Keep custom solution.

---

## Action Items

### Immediate (TEAM-305)

**1. Add tokio-util dependency:**
```toml
[dependencies]
tokio-util = { version = "0.7", features = ["sync", "time"] }
```

**2. Implement job cancellation:**
- Add `CancellationToken` to `Job` struct
- Add `cancel_job()` method
- Update `execute_and_stream()` to support cancellation
- Add tests

**3. Implement job timeout:**
- Add `execute_and_stream_with_timeout()`
- Use `tokio::time::timeout()` wrapper
- Add tests

**Effort:** 2-3 days  
**Impact:** High - Prevents hung jobs and resource exhaustion

---

### Next Sprint

**4. Add retry logic:**
- Add `RetryConfig` struct
- Add `Retrying` state
- Implement exponential backoff
- Add tests

**5. Add priority queue:**
- Add `JobPriority` enum
- Add `get_next_job()` method
- Update job creation
- Add tests

**Effort:** 3-4 days  
**Impact:** Medium - Improves reliability and resource allocation

---

### Future

**6. Evaluate tracing integration:**
- Prototype hybrid approach (tracing + narration-core)
- Create custom SSE subscriber
- Measure performance impact
- Document migration path

**Effort:** 1 week  
**Impact:** Medium - Better backend observability

**7. Consider persistence (only if needed):**
- Evaluate effectum for SQLite-based persistence
- Only if job durability becomes requirement
- Not needed for current use case

---

## Key Decisions

### ✅ Keep Current Structure
- No reorganization into `job-lifecycle/` folder
- File is manageable at 448 LOC
- Clear and simple structure

### ✅ Keep Custom Implementation
- Well-suited for our dual-call + SSE pattern
- No external job queue library fits our needs
- Tight integration with narration-core

### ✅ Adopt tokio-util
- For `CancellationToken` (job cancellation)
- For `DelayQueue` (retry scheduling)
- Minimal dependency, high value

### 🔶 Consider tracing (Future)
- For backend observability
- Hybrid approach with narration-core
- Not urgent, revisit in 6 months

### ❌ Avoid Job Queue Libraries
- apalis, fang, effectum - wrong architectural model
- Designed for worker pools, not dual-call pattern
- Adds unnecessary complexity

---

## References

- **Engineering Rules:** `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`
- **daemon-lifecycle:** `/home/vince/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle/`
- **narration-core:** `/home/vince/Projects/llama-orch/bin/99_shared_crates/narration-core/`
- **TEAM-304 Handoff:** `/home/vince/Projects/llama-orch/bin/99_shared_crates/narration-core/.plan/TEAM_304_HANDOFF.md`

---

## Document History

- **2025-10-26:** Initial analysis by TEAM-304
- **Next Review:** After Phase 1 implementation (timeout + cancellation)

---

**TL;DR:**
- ✅ Keep current structure (no reorganization)
- ✅ Keep custom implementation (fits our needs)
- ✅ Add tokio-util (for cancellation + timeout)
- 🔶 Consider tracing later (for observability)
- ❌ Don't adopt job queue libraries (wrong model)
