# External Crate Evaluation for Job Lifecycle & Observability

**Date:** October 26, 2025  
**Author:** TEAM-304  
**Status:** Research & Recommendations

---

## Executive Summary

**Verdict:** ✅ **Keep our custom implementation, but adopt specific external crates for complementary features**

After extensive research, our custom `job-server` + `narration-core` solution is **appropriate and well-designed**. However, we should adopt external crates for specific capabilities where they add clear value.

---

## Research Methodology

**Evaluated:**
- Job queue libraries (apalis, fang, effectum)
- SSE/streaming libraries (axum-sse, tower-http)
- Observability libraries (tracing, opentelemetry)
- State machine libraries
- Task scheduling libraries

**Criteria:**
1. **Fit:** Does it solve our specific problem?
2. **Complexity:** Does it add unnecessary overhead?
3. **Maintenance:** Is it actively maintained?
4. **Integration:** Does it integrate with our stack (axum, tokio)?
5. **Size:** Is it appropriately scoped?

---

## Category 1: Job Queue / Background Processing

### 1.1 apalis

**Link:** https://github.com/geofmureithi/apalis  
**Stars:** ~1,000  
**Maintenance:** ✅ Active (last commit: recent)

**Features:**
- Full-featured job queue with multiple backends (Redis, PostgreSQL, SQLite, in-memory)
- Cron scheduling support
- Job retry with exponential backoff
- Job priorities
- Middleware support (Tower-based)
- Worker pools

**Pros:**
- ✅ Production-ready
- ✅ Well-documented
- ✅ Multiple storage backends
- ✅ Tower middleware integration

**Cons:**
- ❌ **Overkill** - We don't need persistence or distributed coordination
- ❌ **Heavy** - Adds significant dependencies
- ❌ **Different model** - Uses worker pools, not our dual-call SSE pattern
- ❌ **No SSE integration** - Doesn't fit our streaming architecture

**Verdict:** ❌ **DO NOT ADOPT**

**Reason:** Apalis is designed for traditional background job processing with worker pools. Our architecture is fundamentally different:
- We use **dual-call pattern** (POST creates job, GET streams results)
- We use **SSE streaming** for real-time feedback
- We don't need **persistence** (jobs are ephemeral)
- We don't need **distributed coordination** (single-node for now)

**What we can learn:**
- Retry logic patterns
- Priority queue implementation
- Timeout handling

---

### 1.2 fang

**Link:** https://github.com/ayrat555/fang  
**Stars:** ~500  
**Maintenance:** ✅ Active

**Features:**
- PostgreSQL/SQLite/MySQL backend
- Async and threaded workers
- Scheduled tasks (cron)
- Unique tasks (deduplication)
- Periodic tasks

**Pros:**
- ✅ Simpler than apalis
- ✅ Good async support
- ✅ Database-backed (durable)

**Cons:**
- ❌ **Still overkill** - Requires database
- ❌ **Worker pool model** - Not our architecture
- ❌ **No SSE** - Doesn't fit streaming pattern

**Verdict:** ❌ **DO NOT ADOPT**

**Reason:** Same as apalis - different architectural model. We don't need database-backed job persistence.

---

### 1.3 effectum

**Link:** https://github.com/dimfeld/effectum  
**Stars:** ~200  
**Maintenance:** ✅ Active

**Features:**
- SQLite-based (embedded)
- Lightweight
- Job state machine
- Retry logic
- Scheduled jobs

**Pros:**
- ✅ Lightweight (SQLite embedded)
- ✅ No external dependencies
- ✅ Good for single-node

**Cons:**
- ❌ **Persistence overhead** - We don't need SQLite
- ❌ **Different pattern** - Not SSE-based
- ❌ **Smaller community** - Less battle-tested

**Verdict:** ❌ **DO NOT ADOPT** (but interesting for future)

**Reason:** If we ever need persistence, effectum is a good candidate. But for now, in-memory is sufficient.

**Future consideration:** If we need job durability across restarts, revisit effectum.

---

### 1.4 tokio-cron-scheduler

**Link:** https://crates.io/crates/tokio-cron-scheduler  
**Stars:** ~300  
**Maintenance:** ✅ Active

**Features:**
- Cron-based scheduling
- Async task execution
- Simple API

**Pros:**
- ✅ Simple and focused
- ✅ Good for periodic tasks
- ✅ Tokio-based

**Cons:**
- ❌ **Not needed yet** - We don't have scheduled jobs
- ❌ **Different use case** - Cron vs on-demand jobs

**Verdict:** 🔶 **CONSIDER FOR FUTURE**

**Reason:** If we add scheduled/periodic jobs (e.g., cleanup tasks, health checks), this is a good fit.

**Use case:** Periodic cleanup of old jobs, scheduled model updates, etc.

---

## Category 2: SSE / Streaming

### 2.1 axum::response::sse (Built-in)

**Link:** https://docs.rs/axum/latest/axum/response/sse/  
**Status:** ✅ **ALREADY USING**

**Features:**
- Server-Sent Events support
- Keep-alive
- Event formatting
- Stream integration

**Verdict:** ✅ **KEEP USING**

**Reason:** Built into axum, works perfectly for our use case. No need for external crate.

**What we're doing right:**
- Using `Sse::new(event_stream)` for streaming
- Proper event formatting
- [DONE] and [ERROR] signals at transport layer

---

### 2.2 tower-http (Middleware)

**Link:** https://docs.rs/tower-http/  
**Status:** ✅ **ALREADY USING**

**Features:**
- CORS middleware
- Compression
- Tracing middleware
- Request ID propagation

**Verdict:** ✅ **KEEP USING**

**Reason:** Essential for HTTP middleware. Already integrated.

**Potential enhancement:**
- Add `tower-http::request_id` for automatic request ID generation
- Add `tower-http::trace` for better request tracing

---

## Category 3: Observability / Tracing

### 3.1 tracing + tracing-subscriber

**Link:** https://docs.rs/tracing/  
**Stars:** ~5,000  
**Maintenance:** ✅ Active (Tokio project)

**Features:**
- Structured logging
- Span-based tracing
- Context propagation
- Multiple output formats (JSON, pretty, etc.)
- Integration with OpenTelemetry

**Pros:**
- ✅ **Industry standard** for Rust observability
- ✅ **Structured logging** - Better than println!
- ✅ **Span tracking** - Automatic context propagation
- ✅ **Production-ready** - Used by major projects
- ✅ **Flexible** - Multiple subscribers (stdout, file, Jaeger, etc.)

**Cons:**
- ⚠️ **Different model** - Spans vs our narration events
- ⚠️ **Migration effort** - Would require refactoring narration-core
- ⚠️ **Learning curve** - Team needs to learn tracing concepts

**Verdict:** 🔶 **CONSIDER FOR FUTURE** (Major decision)

**Analysis:**

**Our narration-core:**
```rust
NARRATE
    .action("hive_start")
    .job_id(&job_id)
    .context("localhost")
    .human("Starting hive {}")
    .emit();
```

**With tracing:**
```rust
#[instrument(skip(config), fields(job_id = %job_id))]
async fn start_hive(job_id: &str, config: Arc<Config>) -> Result<()> {
    info!(hive = "localhost", "Starting hive");
    // ...
}
```

**Key differences:**

| Feature | narration-core | tracing |
|---------|----------------|---------|
| **SSE routing** | ✅ Built-in (job_id) | ❌ Requires custom subscriber |
| **Human messages** | ✅ First-class | ⚠️ Via formatting |
| **Structured data** | ✅ context() calls | ✅ Span fields |
| **Ecosystem** | ❌ Custom | ✅ Industry standard |
| **OpenTelemetry** | ❌ No | ✅ Yes |
| **Learning curve** | ✅ Simple | ⚠️ Moderate |

**Recommendation:**

**Option A: Keep narration-core** (Recommended for now)
- ✅ Works well for our use case
- ✅ SSE integration is seamless
- ✅ Team is familiar
- ✅ Simple mental model

**Option B: Migrate to tracing** (Future consideration)
- ✅ Industry standard
- ✅ Better ecosystem
- ✅ OpenTelemetry support
- ❌ Requires significant refactoring
- ❌ Need custom SSE subscriber

**Hybrid approach:** Use both
- `tracing` for internal spans and structured logging
- `narration-core` for SSE streaming to clients
- Bridge between them via custom subscriber

**Decision:** ✅ **Keep narration-core for now, revisit tracing in 6 months**

**Reason:** Our narration-core is well-suited for SSE streaming. Tracing is better for backend observability. We can use both if needed.

---

### 3.2 opentelemetry-rust

**Link:** https://github.com/open-telemetry/opentelemetry-rust  
**Stars:** ~1,500  
**Maintenance:** ✅ Active

**Features:**
- Distributed tracing
- Metrics collection
- Log correlation
- Export to Jaeger, Prometheus, etc.

**Verdict:** 🔷 **LOW PRIORITY** (Future)

**Reason:** OpenTelemetry is for distributed systems. We're single-node for now. Revisit when we scale horizontally.

---

## Category 4: State Machine

### 4.1 state_machine_future

**Link:** https://crates.io/crates/state_machine_future  
**Maintenance:** ⚠️ Archived

**Verdict:** ❌ **DO NOT USE** (Archived)

---

### 4.2 smlang (State Machine Language)

**Link:** https://crates.io/crates/smlang  
**Stars:** ~200  
**Maintenance:** ✅ Active

**Features:**
- Compile-time state machine generation
- Type-safe transitions
- Zero-cost abstractions

**Example:**
```rust
state_machine! {
    JobStateMachine(Queued)

    Queued => Running,
    Running => Completed,
    Running => Failed,
}
```

**Verdict:** 🔷 **LOW PRIORITY** (Nice to have)

**Reason:** Our state machine is simple (4 states, 5 transitions). A macro-based solution is overkill. But if we add more states (Retrying, Cancelled, Paused), revisit this.

**Current approach is fine:**
```rust
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed(String),
}
```

---

## Category 5: Async Utilities

### 5.1 tokio-util

**Link:** https://docs.rs/tokio-util/  
**Status:** ✅ **SHOULD ADOPT**

**Features:**
- `CancellationToken` - For job cancellation
- `Codec` - For framing
- `ReusableBoxFuture` - For optimization
- `time::DelayQueue` - For scheduling

**Verdict:** ✅ **ADOPT** (Specific features)

**Recommended usage:**

**1. CancellationToken for job cancellation:**
```rust
use tokio_util::sync::CancellationToken;

pub struct Job<T> {
    // ... existing fields ...
    pub cancellation_token: CancellationToken,
}

// In executor:
tokio::select! {
    result = executor(job_id, payload) => { /* ... */ }
    _ = cancellation_token.cancelled() => {
        // Job cancelled
    }
}
```

**2. DelayQueue for job scheduling:**
```rust
use tokio_util::time::DelayQueue;

// For retry logic:
let mut delay_queue = DelayQueue::new();
delay_queue.insert(job_id, retry_delay);
```

**Action:** ✅ **Add tokio-util to Cargo.toml**

---

### 5.2 futures-util

**Link:** https://docs.rs/futures-util/  
**Status:** ✅ **ALREADY USING**

**Verdict:** ✅ **KEEP USING**

**Reason:** Essential for stream utilities. Already integrated.

---

## Category 6: HTTP Client

### 6.1 reqwest

**Link:** https://docs.rs/reqwest/  
**Status:** ✅ **ALREADY USING**

**Verdict:** ✅ **KEEP USING**

**Reason:** Industry standard HTTP client. Works well for our use case (job-client, hive-forwarder).

**What we're doing right:**
- Using `reqwest::Client` for HTTP requests
- Proper error handling
- SSE streaming via `bytes_stream()`

---

## Recommendations Summary

### ✅ Adopt Immediately

1. **tokio-util** (specific features)
   - `CancellationToken` for job cancellation
   - `DelayQueue` for retry scheduling
   - **Effort:** 1 day
   - **Impact:** High

### 🔶 Consider for Next Phase

2. **tokio-cron-scheduler** (if we add scheduled jobs)
   - For periodic cleanup tasks
   - For scheduled model updates
   - **Effort:** 2 days
   - **Impact:** Medium

3. **tracing + tracing-subscriber** (major decision)
   - For backend observability
   - Hybrid approach with narration-core
   - **Effort:** 1-2 weeks
   - **Impact:** High (but not urgent)

### 🔷 Future Consideration

4. **effectum** (if we need persistence)
   - Only if job durability becomes requirement
   - **Effort:** 3-4 days
   - **Impact:** Low (not needed now)

5. **opentelemetry-rust** (if we scale horizontally)
   - For distributed tracing
   - **Effort:** 1-2 weeks
   - **Impact:** Low (single-node for now)

### ❌ Do Not Adopt

6. **apalis, fang** - Wrong architectural model
7. **state_machine_future** - Archived
8. **smlang** - Overkill for our simple state machine

---

## Specific Action Items

### Immediate (TEAM-305)

**1. Add tokio-util to job-server:**

```toml
# Cargo.toml
[dependencies]
tokio-util = { version = "0.7", features = ["sync", "time"] }
```

**2. Implement CancellationToken:**
- Add to `Job` struct
- Update `execute_and_stream()` to support cancellation
- Add `cancel_job()` method to `JobRegistry`
- Add tests

**3. Implement timeout with tokio::time:**
- Add `execute_and_stream_with_timeout()`
- Use `tokio::time::timeout()` wrapper
- Add tests

**Effort:** 2-3 days  
**Impact:** High - Prevents hung jobs

---

### Next Sprint

**4. Evaluate tracing integration:**
- Prototype hybrid approach (tracing + narration-core)
- Create custom SSE subscriber for tracing
- Measure performance impact
- Document migration path

**Effort:** 1 week  
**Impact:** Medium - Better observability

---

### Future

**5. Monitor external crate ecosystem:**
- Check for new job queue libraries
- Watch for SSE/streaming improvements in axum
- Track OpenTelemetry adoption in Rust

**Effort:** Ongoing  
**Impact:** Low - Stay informed

---

## Conclusion

### Summary

**Our custom implementation is appropriate:**
- ✅ Well-designed for our use case (dual-call + SSE)
- ✅ Simple and maintainable
- ✅ No unnecessary dependencies
- ✅ Tight integration with our stack

**External crates to adopt:**
- ✅ **tokio-util** - For cancellation and scheduling
- 🔶 **tracing** - Consider for backend observability (not urgent)

**External crates to avoid:**
- ❌ **apalis, fang, effectum** - Wrong architectural model
- ❌ **State machine libraries** - Overkill for our simple state machine

### Key Insight

**Don't adopt external crates just because they exist.** Our custom solution is well-suited for our specific requirements:
1. **Dual-call pattern** (POST + GET)
2. **SSE streaming** (real-time feedback)
3. **Ephemeral jobs** (no persistence needed)
4. **Single-node** (no distributed coordination)

External job queue libraries (apalis, fang, effectum) are designed for different use cases:
- Worker pools (not dual-call)
- Database persistence (not ephemeral)
- Traditional background jobs (not SSE streaming)

**Our architecture is unique and appropriate. Keep it.**

### Final Recommendation

1. **Keep job-server as is** - Well-designed, appropriate scope
2. **Add tokio-util** - For cancellation and timeout
3. **Consider tracing** - For backend observability (6-month review)
4. **Avoid job queue libraries** - Wrong architectural model

**Don't fix what isn't broken. Enhance what works.**

---

**Document Version:** 1.0  
**Last Updated:** October 26, 2025  
**Next Review:** After tokio-util integration
