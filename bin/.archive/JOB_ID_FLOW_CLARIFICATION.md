# Job ID Flow - The ACTUAL Architecture

**Date:** 2025-10-22 03:55 AM  
**Status:** 🔍 CLARIFICATION for tired brain

---

## Your Confusion is Valid!

You asked: "Is each component making a new job_id? Seems redundant."

**Answer: NO!** Only ONE job_id per user request (at Queen level).

---

## The ACTUAL Flow

```
┌─────────────────────────────────────────────────────────────┐
│ User Request: ./rbee hive start                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ KEEPER (CLI)                                                │
│ - No job_id yet                                             │
│ - POST /v1/jobs { operation: "hive_start", hive_id: "..." }│
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ QUEEN                                                       │
│ - Creates job_id = "job-abc123" ← ONLY ONE                 │
│ - Returns: { job_id: "job-abc123", sse_url: "/stream" }    │
│ - Keeper connects to SSE: GET /v1/jobs/job-abc123/stream   │
│                                                             │
│ ALL NARRATIONS FROM QUEEN USE job-abc123:                  │
│   NARRATE.action("hive_start").job_id("job-abc123").emit() │
│   NARRATE.action("hive_check").job_id("job-abc123").emit() │
│   NARRATE.action("hive_spawn").job_id("job-abc123").emit() │
│                                                             │
│ ✅ These all go to keeper via SSE (same job_id)            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ HIVE (spawned as daemon)                                    │
│ - Queen spawns it via SSH/local                             │
│ - NO NEW JOB created!                                       │
│ - Hive just runs as a daemon                                │
│ - Hive has its OWN endpoints for when IT gets requests      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ WORKER (spawned by hive)                                    │
│ - Hive spawns worker process                                │
│ - NO NEW JOB in queen's context!                            │
│ - Worker is just a process                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## When Does Each Component Create a Job?

### Queen Creates Jobs

**ONLY when a client makes a request:**

```
Keeper → POST /v1/jobs → Queen creates job-abc123
```

All operations within that request use `job-abc123` for narration.

### Hive Creates Jobs

**ONLY when IT receives a direct request:**

```
Some Client → POST http://hive:9000/v1/jobs → Hive creates job-def456
```

This is a SEPARATE job, not related to queen's job.

### Worker Creates Jobs

**NEVER!** Workers don't have the job-based architecture (yet?).

---

## The Confusion: Inference Flow

You said: "The queen connects to the worker directly during inference"

**CORRECT!** For inference:

```
Keeper → POST /v1/jobs (operation: "infer") → Queen (job-abc123)
  ↓
Queen finds available worker
  ↓
Queen → HTTP request → Worker (inference request)
  ↓
Worker streams tokens back to Queen
  ↓
Queen streams tokens to Keeper via SSE (job-abc123)
```

**ONE job_id throughout!** (`job-abc123` at Queen level)

---

## Why Each Service Has job_id in Narration

### Queen's Narrations

```rust
// All use job-abc123 (the request from keeper)
NARRATE.action("route_job").job_id("job-abc123").emit();
NARRATE.action("hive_start").job_id("job-abc123").emit();
NARRATE.action("worker_spawn").job_id("job-abc123").emit();
```

**Why?** So keeper receives them via SSE on `/v1/jobs/job-abc123/stream`

### Hive's Narrations (if hive gets a direct request)

```rust
// Different job! This is when someone calls hive directly
NARRATE.action("worker_spawn").job_id("job-def456").emit();
```

**Why?** So the hive's client receives them via SSE on `http://hive:9000/v1/jobs/job-def456/stream`

---

## The Pattern is NOT Redundant!

### Each SERVICE can receive job requests:

```
Keeper → Queen (creates job-abc123)
Some Tool → Hive (creates job-def456)  
Some Script → Worker (no job architecture yet)
```

Each service needs its own job registry because each CAN be called directly.

### But for a SINGLE user request:

```
Keeper → Queen (job-abc123)
  ↓
  Queen does everything in context of job-abc123
  ↓
  All narrations use job-abc123
  ↓
  Keeper receives all events via SSE
```

**ONE job_id for the entire request!** ✅

---

## Your Question About Distribution

> "Can we not get ONE job id and the data gets distributed first, then the execution comes when all the SSE connects..."

**You're describing the CURRENT system!**

1. Keeper creates job → Queen generates job-abc123
2. Keeper connects to SSE → GET /v1/jobs/job-abc123/stream
3. **THEN** execution starts (Queen sees SSE connection, starts processing)
4. All narrations from Queen use job-abc123
5. All flow to keeper's SSE connection

---

## The Real Question: Why Repetition?

### In Queen's code:

```rust
async fn route_operation(job_id: String, ...) {
    // Why repeat .job_id(&job_id) everywhere?
    NARRATE.action("hive_start").job_id(&job_id).emit();
    NARRATE.action("hive_check").job_id(&job_id).emit();
    NARRATE.action("hive_spawn").job_id(&job_id).emit();
}
```

**Answer:** SSE sink requires it for routing. Without it, events are dropped.

**It's the SAME job_id** (from the function parameter), just repeated for each narration.

---

## What About Correlation ID?

**Correlation ID is for end-to-end tracing:**

```
Keeper (no corr_id yet)
  ↓
Queen generates: correlation_id = "corr-xyz789"
  ↓
If Queen calls Hive: passes correlation_id
  ↓
If Hive calls Worker: passes correlation_id
  ↓
All logs can be searched by correlation_id
```

**This is for TRACING, not for SSE routing.**

---

## Summary for Tired Brain 😴

1. **ONE job_id per user request** (created at Queen)
2. **All narrations in Queen use that SAME job_id** (hence the repetition)
3. **SSE routing requires job_id** (can't eliminate repetition)
4. **Each service CAN create jobs** (when called directly), but in a single request flow, there's only one
5. **Correlation ID is different** (for end-to-end tracing across services)

---

## Go to Sleep! 🌙

The architecture is correct. The repetition is necessary. Don't overthink it when tired!

Sweet dreams! 😴
