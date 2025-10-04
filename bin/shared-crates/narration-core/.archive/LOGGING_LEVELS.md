# 🎚️ Narration Logging Levels Specification

**Owner**: Narration Core Team  
**Status**: Approved ✅  
**Version**: 1.0.0  
**Decision**: Build Our Own Custom Narration System 🎀

---

## 📋 Overview

The narration system supports **7 distinct logging levels** (MUTE + 6 operational levels) to control verbosity and observability granularity. Each level has **clear boundaries** and **specific use cases**.

**Our Custom Implementation**:
- 🎀 Custom proc macros (`#[trace_fn]`, `#[narrate(...)]`)
- 🎭 Auto-inferred actor from module path
- 🎨 Template interpolation for human/cute/story fields
- 🔒 Compile-time editorial enforcement (≤100 chars, SVO validation)
- 📊 Conditional compilation (zero overhead in production)
- 💝 Cute/story modes built-in (first-class, not add-on)

This document defines:
- **What belongs at each level**
- **What does NOT belong at each level**
- **When to use each level**
- **Examples for each level** (with cute mode!)
- **Editorial guidelines per level**
- **Integration with our custom proc macros**

---

## 🎯 The Seven Levels

| Level | Purpose | Visibility | Production Use | Development Use |
|-------|---------|------------|----------------|-----------------|
| **MUTE** | Complete silence | Nothing | Critical security contexts | Never |
| **TRACE** | Exhaustive detail | Every single event | ❌ Never | ⚠️ Targeted debugging |
| **DEBUG** | Developer debugging | Detailed flow + context | ⚠️ Incidents only | ✅ Always |
| **INFO** | Production-ready stories | Key lifecycle events | ✅ Always | ✅ Default |
| **WARN** | Anomalies & degradations | Actionable warnings | ✅ Always | ✅ Always |
| **ERROR** | Operational failures | Rich context | ✅ Always | ✅ Always |
| **FATAL** | Unrecoverable errors | Maximum context | ✅ Always | ✅ Always |

---

## 🔇 Level 1: MUTE

### Definition
**Complete silence**. No narration events are emitted at all.

### When to Use
- **Security-critical contexts** where any logging could leak sensitive information
- **High-frequency hot paths** where even INFO would cause performance degradation
- **Compliance requirements** that mandate zero logging in specific code paths
- **Temporary suppression** during sensitive operations

### What Belongs Here
**NOTHING.** Mute means mute. No events, no stories, no logs.

### What Does NOT Belong Here
- Normal production operations (use INFO)
- Debugging (use DEBUG or TRACE)
- Error reporting (use INFO with redaction)

### Implementation
```rust
// Set narration policy to MUTE
NarrationPolicy::set_level(NarrationLevel::Mute);

// All narration calls are now no-ops
narrate(NarrationFields {
    actor: "orchestratord",
    action: "process",
    target: "secret-job".to_string(),
    human: "Processing sensitive data".to_string(),
    ..Default::default()
}); // This emits NOTHING
```

### Editorial Guidelines
- ✅ **Use sparingly**: Only for genuine security/performance needs
- ❌ **Never use as default**: INFO should be your production baseline
- ⚠️ **Document why**: Always comment why MUTE is necessary

### Examples

**Valid MUTE usage**:
```rust
// MUTE: PCI-DSS compliance requires zero logging during card processing
NarrationPolicy::set_level(NarrationLevel::Mute);
process_credit_card_transaction(card_data).await?;
NarrationPolicy::set_level(NarrationLevel::Info);
```

**Invalid MUTE usage**:
```rust
// ❌ WRONG: Don't use MUTE to hide bugs
NarrationPolicy::set_level(NarrationLevel::Mute);
buggy_function_that_logs_too_much().await?;
```

---

## 📢 Level 2: INFO

### Definition
**Production-ready stories**. Key lifecycle events that tell the system's story without overwhelming operators.

### When to Use
- **Production deployments** (default level)
- **Key lifecycle events** (startup, shutdown, job acceptance, completion)
- **Important state changes** (worker ready, pool registered, model loaded)
- **User-facing operations** (request accepted, job dispatched, result returned)
- **Errors and warnings** (failures, retries, degraded states)

### What Belongs Here
- ✅ **Request lifecycle**: Accepted, queued, dispatched, completed
- ✅ **Resource lifecycle**: Worker ready, pool registered, engine spawned
- ✅ **State transitions**: Idle → busy, healthy → degraded
- ✅ **Errors**: All errors (with redaction)
- ✅ **Performance milestones**: Job completed in X ms, Y tokens generated
- ✅ **Configuration changes**: Policy updated, pool added/removed

### What Does NOT Belong Here
- ❌ **Internal implementation details**: Loop iterations, intermediate calculations
- ❌ **High-frequency events**: Heartbeat ticks (unless they fail)
- ❌ **Verbose debugging**: Full request/response payloads
- ❌ **Trace-level granularity**: Every function call

### Implementation
```rust
// INFO is the default level
narrate(NarrationFields {
    actor: "orchestratord",
    action: "accept",
    target: "job-123".to_string(),
    human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'".to_string(),
    correlation_id: Some("req-abc".into()),
    queue_position: Some(3),
    predicted_start_ms: Some(420),
    pool_id: Some("default".into()),
    ..Default::default()
});
```

### Editorial Guidelines
- ✅ **Tell the story**: Each event should advance the narrative
- ✅ **Be specific**: Include IDs, numbers, context
- ✅ **Stay concise**: ≤100 characters (ORCH-3305)
- ✅ **Use present tense**: "Accepting request" not "Accepted request" (for in-progress)
- ✅ **Include correlation IDs**: Always propagate request tracking
- ❌ **Avoid noise**: Don't log every internal step

### Examples

**Perfect INFO narrations**:
```rust
// Job acceptance
human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"

// Worker ready
human: "Worker ready with engine llamacpp-v1, 8 slots available"

// Job completion
human: "Job completed in 2500 ms (150 tokens, 60 tokens/sec)"

// Error with context
human: "VRAM allocation failed: requested 4096 MB, only 2048 MB available on GPU0"

// State transition
human: "Pool 'default' transitioned from healthy to degraded (2/3 workers alive)"
```

**Bad INFO narrations** (too verbose):
```rust
// ❌ TOO DETAILED: This belongs in DEBUG
human: "Entering dispatch_job function with job_id=123, pool_id=default, correlation_id=req-abc"

// ❌ TOO FREQUENT: This belongs in TRACE
human: "Heartbeat tick 1234 from worker-gpu0-r1"

// ❌ TOO VAGUE: Needs more context
human: "Error occurred"
```

---

## ⚠️ Level 3: WARN

### Definition
**Anomalies & degradations**. Actionable warnings about non-fatal issues that need attention.

### When to Use
- **Retry attempts** (job retries, connection retries)
- **Deprecated features** (flags, APIs, configurations)
- **Performance degradation** (slow workers, high latency)
- **Resource pressure** (queue near capacity, memory pressure)
- **Non-fatal errors** (missed heartbeats, transient failures)

### What Belongs Here
- ✅ **Retry attempts**: "Retrying job-123 after timeout (attempt 2/5)"
- ✅ **Deprecated warnings**: "Using deprecated flag --legacy-mode"
- ✅ **Performance issues**: "Worker latency: 500ms (expected <100ms)"
- ✅ **Resource warnings**: "Queue depth: 95/100 (95% capacity)"
- ✅ **Transient failures**: "Heartbeat missed (last seen 5000ms ago)"

### What Does NOT Belong Here
- ❌ **Fatal errors**: Use ERROR or FATAL
- ❌ **Normal operations**: Use INFO
- ❌ **Debug details**: Use DEBUG

### Implementation
```rust
// WARN-level narration
narrate_warn(NarrationFields {
    actor: "orchestratord",
    action: "retry",
    target: "job-123".to_string(),
    human: "Retrying job-123 after timeout (attempt 2/5, backoff: 200ms)".to_string(),
    correlation_id: Some("req-abc".into()),
    retry_attempt: Some(2),
    max_retries: Some(5),
    backoff_ms: Some(200),
    ..Default::default()
});
```

### Editorial Guidelines
- 📏 Length: ≤120 characters (need room for context)
- 🔗 Correlation: Always propagate
- 🎯 Actionability: Include what's wrong AND what to do
- ⚠️ Severity: Not fatal, but needs attention
- 🎀 Cute mode: Gentle, concerned tone

### Examples

**Perfect WARN narrations**:
```rust
// Retry attempt
human: "Retrying job-123 after timeout (attempt 2/5, backoff: 200ms)"
cute: "Job-123 didn't make it through. Let's try again! Attempt 2... 🔄"

// Deprecated flag
human: "Using deprecated flag --legacy-mode; migrate to --compat-mode by v2.0"
cute: "Psst! --legacy-mode is getting old. Switch to --compat-mode soon! 📢"

// Performance degradation
human: "Worker-gpu0-r1 latency: 500ms (expected <100ms, threshold exceeded)"
cute: "Worker-gpu0-r1 is running a bit slow today (500ms). Might need a checkup! 🐌"

// Resource pressure
human: "Queue depth: 95/100 (95% capacity, consider scaling)"
cute: "The queue is getting pretty full! 95 jobs waiting! Maybe add more workers? 📈"

// Missed heartbeat
human: "Heartbeat missed from worker-gpu0-r1 (last seen 5000ms ago, threshold: 3000ms)"
cute: "Haven't heard from worker-gpu0-r1 in 5 seconds. Are they okay? 😟"
```

---

## ❌ Level 4: ERROR

### Definition
**Operational failures**. Rich context for diagnosing failures that prevent operations from completing.

### When to Use
- **Failed operations** (VRAM allocation failed, connection refused)
- **Validation failures** (invalid model, missing config)
- **Network errors** (timeout, connection refused, DNS failure)
- **Resource exhaustion** (out of memory, disk full)
- **Timeout errors** (job timeout, request timeout)

### What Belongs Here
- ✅ **Failed operations with context**: "VRAM allocation failed: requested 4096MB, only 2048MB available"
- ✅ **Validation failures**: "Job validation failed: model 'invalid-model' not found"
- ✅ **Network errors**: "Failed to connect to pool-managerd: connection refused"
- ✅ **Resource exhaustion**: "GPU0 out of memory: 8192MB allocated, 0MB free"
- ✅ **Timeout errors**: "Job-123 timed out after 30s (max: 30s)"

### What Does NOT Belong Here
- ❌ **Unrecoverable errors**: Use FATAL
- ❌ **Warnings**: Use WARN
- ❌ **Normal failures that retry**: Use WARN

### Implementation
```rust
// ERROR-level narration
narrate_error(NarrationFields {
    actor: "pool-managerd",
    action: "allocate_vram",
    target: "GPU0".to_string(),
    human: "VRAM allocation failed on GPU0: requested 4096MB, only 2048MB available".to_string(),
    correlation_id: Some("req-abc".into()),
    error_kind: Some("vram_exhaustion".into()),
    requested_mb: Some(4096),
    available_mb: Some(2048),
    device: Some("GPU0".into()),
    ..Default::default()
});
```

### Editorial Guidelines
- 📏 Length: ≤150 characters (need room for diagnosis)
- 🔗 Correlation: **ALWAYS** propagate (critical for debugging!)
- 🎯 Diagnosis: Include what failed, why, and relevant context
- 🔒 Redaction: Especially important for error details
- 🎀 Cute mode: Empathetic, helpful tone

### Examples

**Perfect ERROR narrations**:
```rust
// VRAM allocation failure
human: "VRAM allocation failed on GPU0: requested 4096MB, only 2048MB available"
cute: "Oh no! GPU0 doesn't have enough room (need 4GB, only 2GB free). 😟"
story: "\"Do you have 4GB?\" asked orchestratord. \"No,\" replied GPU0 sadly, \"only 2GB free.\""

// Validation failure
human: "Job validation failed: model 'gpt-5' not found in catalog (available: llama-7b, phi-3)"
cute: "Hmm, 'gpt-5' isn't in our catalog. We have llama-7b and phi-3 though! 🔍"

// Network failure
human: "Failed to connect to pool-managerd at localhost:8080: connection refused (retry in 1s)"
cute: "Couldn't reach pool-managerd! They might be napping. Trying again in 1s... 😴"

// Resource exhaustion
human: "GPU0 out of memory: 8192MB allocated, 0MB free (cannot load model 'llama-13b')"
cute: "GPU0 is completely full! No room for llama-13b right now. 😰"

// Timeout
human: "Job-123 timed out after 30s (max: 30s, worker: worker-gpu0-r1, correlation: req-abc)"
cute: "Job-123 took too long (30s limit). Worker-gpu0-r1 might be stuck! ⏰"
```

---

## 🚨 Level 5: FATAL

### Definition
**Unrecoverable errors**. Maximum context for panic-worthy failures that require immediate shutdown.

### When to Use
- **Policy violations** (VRAM-only policy violated, security policy breached)
- **Data corruption** (seal verification failed, checksum mismatch)
- **Invariant violations** (duplicate IDs, corrupted registry)
- **Total resource failure** (all GPUs failed, no workers available)
- **Security violations** (unauthorized access, token theft)

### What Belongs Here
- ✅ **Policy violations**: "CRITICAL: VRAM-only policy violated on GPU0: UMA detected"
- ✅ **Data corruption**: "CRITICAL: Seal verification failed: digest mismatch"
- ✅ **Invariant violations**: "CRITICAL: Worker registry corrupted: duplicate worker_id"
- ✅ **Total failures**: "CRITICAL: All GPUs failed health check"
- ✅ **Security violations**: "CRITICAL: Unauthorized access attempt detected"

### What Does NOT Belong Here
- ❌ **Recoverable errors**: Use ERROR
- ❌ **Warnings**: Use WARN
- ❌ **Normal failures**: Use ERROR

### Implementation
```rust
// FATAL-level narration
narrate_fatal(NarrationFields {
    actor: "worker-orcd",
    action: "policy_violation",
    target: "GPU0".to_string(),
    human: "CRITICAL: VRAM-only policy violated on GPU0: UMA detected. Worker startup aborted.".to_string(),
    correlation_id: Some("req-abc".into()),
    error_kind: Some("policy_violation".into()),
    policy: Some("vram_only".into()),
    device: Some("GPU0".into()),
    reason: Some("UMA detected".into()),
    ..Default::default()
});
```

### Editorial Guidelines
- 📏 Length: Unlimited (this is a crisis, tell the whole story!)
- 🔗 Correlation: **ALWAYS** propagate
- 🎯 Context: Include EVERYTHING (state, inputs, stack trace context)
- 🚨 Severity: System cannot continue
- 🎀 Cute mode: Still gentle, but serious

### Examples

**Perfect FATAL narrations**:
```rust
// Policy violation
human: "CRITICAL: VRAM-only policy violated on GPU0: UMA detected. Worker startup aborted."
cute: "STOP! GPU0 shares memory with CPU (UMA) — we need dedicated VRAM! Shutting down. 🛑"
story: "\"UMA detected!\" cried worker. \"We can't continue,\" said orchestratord gravely. \"Abort.\""

// Data corruption
human: "CRITICAL: Seal verification failed for shard 'llama-7b' on GPU0: digest mismatch (expected: abc123, got: def456)"
cute: "DANGER! llama-7b's safety seal is wrong! This could be corruption! Stopping everything! 🚨"

// Invariant violation
human: "CRITICAL: Worker registry corrupted: duplicate worker_id 'worker-gpu0-r1' (existing: GPU0, new: GPU1)"
cute: "PANIC! Two workers claim to be 'worker-gpu0-r1'! This should never happen! 😱"

// Total resource failure
human: "CRITICAL: All GPUs failed health check (GPU0: OOM, GPU1: driver error, GPU2: offline). Cannot continue."
cute: "All GPUs are down! GPU0 is out of memory, GPU1 has driver issues, GPU2 is offline. We can't run! 💔"

// Security violation
human: "CRITICAL: Unauthorized access attempt from IP 192.168.1.100 (invalid token). Shutting down for security."
cute: "SECURITY ALERT! Someone tried to access us without permission! Shutting down to stay safe! 🔒"
```

---

## 🔍 Level 6: DEBUG

### Definition
**Developer debugging**. Detailed flow and context for understanding system behavior during development and incident response.

### When to Use
- **Local development** (common default for developers)
- **Incident investigation** (temporarily enable in production)
- **Integration debugging** (understanding cross-service flows)
- **Performance analysis** (detailed timing breakdowns)

### What Belongs Here
- ✅ **Function entry/exit**: "Entering dispatch_job with job_id=123"
- ✅ **Decision points**: "Choosing pool 'default' (3 workers available, 1 busy)"
- ✅ **Intermediate state**: "Queue depth: 5, processing capacity: 8"
- ✅ **Retry logic**: "Retry attempt 2/5 after 200 ms backoff"
- ✅ **Cache hits/misses**: "Model cache hit for 'llama-7b' on GPU0"
- ✅ **Detailed timing**: "VRAM allocation took 15 ms (seal: 10 ms, verify: 5 ms)"
- ✅ **Configuration details**: "Using policy: vram_only=true, max_retries=5"

### What Does NOT Belong Here
- ❌ **Every single line**: That's TRACE territory
- ❌ **Sensitive data**: Use redaction or MUTE
- ❌ **Production noise**: Keep INFO clean for operators

### Implementation
```rust
// DEBUG-level narration
narrate_debug(NarrationFields {
    actor: "orchestratord",
    action: "select_pool",
    target: "default".to_string(),
    human: "Choosing pool 'default' (3 workers available, 1 busy, 2 idle)".to_string(),
    correlation_id: Some("req-abc".into()),
    pool_id: Some("default".into()),
    ..Default::default()
});
```

### Editorial Guidelines
- ✅ **Show your work**: Include decision rationale
- ✅ **Add context**: Why did this happen? What were the inputs?
- ✅ **Use structured fields**: Don't just dump strings
- ✅ **Keep correlation IDs**: Still track requests
- ⚠️ **Be verbose but purposeful**: Every DEBUG event should help debugging

### Examples

**Perfect DEBUG narrations**:
```rust
// Function entry with context
human: "Entering dispatch_job: job_id=job-123, pool_id=default, slots_needed=1"

// Decision point
human: "Selected worker-gpu0-r1 (load: 2/8, latency: 50ms, VRAM: 4GB free)"

// Retry logic
human: "Retry attempt 2/5 for job-123 after 200ms backoff (previous error: timeout)"

// Cache behavior
human: "Model cache miss for 'llama-7b'; loading from disk (estimated 500ms)"

// Detailed timing
human: "Job dispatch breakdown: queue_wait=50ms, worker_select=10ms, handoff=5ms"

// Configuration
human: "Applying admission policy: max_queue_depth=100, eta_threshold_ms=5000"
```

**Bad DEBUG narrations**:
```rust
// ❌ TOO VAGUE: Needs more detail
human: "Doing something with job"

// ❌ TOO NOISY: This is TRACE
human: "Loop iteration 47 of 100"

// ❌ WRONG LEVEL: This is INFO
human: "Job completed successfully"
```

---

## 🔬 Level 7: TRACE

### Definition
**Exhaustive detail**. Every single event, every loop iteration, every internal state change. Maximum verbosity.

### When to Use
- **Deep debugging** of specific code paths
- **Performance profiling** with detailed traces
- **Understanding complex algorithms** (e.g., backpressure, scheduling)
- **Reproducing rare bugs** that need complete context

### What Belongs Here
- ✅ **Every function call**: Entry and exit
- ✅ **Every loop iteration**: "Processing item 47/100"
- ✅ **Every state change**: "Queue depth changed from 5 to 6"
- ✅ **Every lock acquisition**: "Acquired mutex for worker registry"
- ✅ **Every validation step**: "Validating field 'pool_id': OK"
- ✅ **Raw data dumps**: Full request/response payloads (redacted)

### What Does NOT Belong Here
- ❌ **Production use**: TRACE will overwhelm your logs
- ❌ **Secrets**: Even TRACE must redact sensitive data

### Implementation
```rust
// TRACE-level narration
narrate_trace(NarrationFields {
    actor: "orchestratord",
    action: "validate",
    target: "job-123".to_string(),
    human: "Validating job field 'model_ref': value='llama-7b', result=OK".to_string(),
    correlation_id: Some("req-abc".into()),
    ..Default::default()
});
```

### Editorial Guidelines
- ✅ **Log everything**: No detail is too small
- ✅ **Include raw data**: Full payloads (redacted)
- ✅ **Show internal state**: Mutex locks, queue depths, counters
- ⚠️ **Performance cost**: TRACE is expensive, use sparingly
- ⚠️ **Still redact secrets**: TRACE ≠ insecure

### Examples

**Perfect TRACE narrations**:
```rust
// Function entry
human: "ENTER dispatch_job(job_id=job-123, pool_id=default, correlation_id=req-abc)"

// Loop iteration
human: "Processing worker 3/8: worker-gpu2-r1 (status=idle, slots=8/8)"

// State change
human: "Queue depth transition: 5 → 6 (added job-123)"

// Lock acquisition
human: "Acquired worker_registry mutex (wait_time=2ms)"

// Validation step
human: "Validating job.model_ref: value='llama-7b', exists=true, compatible=true"

// Raw payload
human: "Received request payload: {\"model\":\"llama-7b\",\"prompt\":\"[REDACTED]\"}"

// Exit
human: "EXIT dispatch_job → Result::Ok(worker-gpu0-r1) (total_time=15ms)"
```

**Bad TRACE narrations**:
```rust
// ❌ TOO VAGUE: Even TRACE needs context
human: "Thing happened"

// ❌ LEAKED SECRET: Always redact
human: "Auth header: Bearer sk-abc123def456"
```

---

## 🚦 Level Selection Guide

### Decision Tree

```
┌─────────────────────────────────────┐
│ Do you need ANY logging?            │
│ NO → MUTE                            │
│ YES ↓                                │
└─────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ Is this an error or failure?        │
│ YES ↓                                │
│ NO → Continue                        │
└─────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ Is it unrecoverable/panic-worthy?   │
│ YES → FATAL                          │
│ NO ↓                                 │
└─────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ Is it a complete failure?           │
│ YES → ERROR                          │
│ NO → WARN (degradation/retry)       │
└─────────────────────────────────────┘

           ↓ (No error)
           
┌─────────────────────────────────────┐
│ Is this a production deployment?    │
│ YES → INFO (default)                 │
│ NO ↓                                 │
└─────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ Are you debugging a specific issue? │
│ YES → DEBUG                          │
│ NO ↓                                 │
└─────────────────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│ Do you need EVERY detail?           │
│ YES → TRACE                          │
│ NO → DEBUG                           │
└─────────────────────────────────────┘
```

### Quick Reference

| Scenario | Level | Rationale |
|----------|-------|-----------|
| Production deployment | **INFO** | Clean, actionable stories |
| Operation failed | **ERROR** | Rich diagnostic context |
| Unrecoverable failure | **FATAL** | Maximum context before shutdown |
| Retry/degradation | **WARN** | Actionable warnings |
| Local development | **DEBUG** | Detailed flow for understanding |
| Incident investigation | **DEBUG** → **TRACE** | Start DEBUG, escalate to TRACE if needed |
| Performance profiling | **TRACE** | Need every timing detail |
| Security audit | **MUTE** → **INFO** | MUTE during sensitive ops, INFO otherwise |
| Integration testing | **DEBUG** | See cross-service flows |
| Unit testing | **TRACE** | Verify every internal step |
| PCI-DSS compliance | **MUTE** | Zero logging during card processing |

---

## 🎨 Editorial Standards Per Level

### INFO Editorial Checklist
- [ ] **Clarity**: Can an operator understand what happened?
- [ ] **Specificity**: Are all relevant IDs/numbers included?
- [ ] **Brevity**: Is it ≤100 characters?
- [ ] **Tense**: Present tense for in-progress, past for completed?
- [ ] **Voice**: Active voice (subject-verb-object)?
- [ ] **Context**: Does it answer "why" not just "what"?
- [ ] **Secrets**: No bearer tokens, API keys, or passwords?
- [ ] **Correlation**: Is correlation_id propagated?

### WARN Editorial Checklist
- [ ] **Actionability**: What's wrong AND what to do about it?
- [ ] **Severity**: Is this truly non-fatal? (Otherwise use ERROR)
- [ ] **Context**: Include relevant metrics (attempt count, threshold, etc.)
- [ ] **Brevity**: Is it ≤120 characters?
- [ ] **Correlation**: Is correlation_id propagated?
- [ ] **Tone**: Gentle but concerned?

### ERROR Editorial Checklist
- [ ] **Diagnosis**: What failed, why, and what was attempted?
- [ ] **Context**: All relevant IDs, metrics, and state included?
- [ ] **Recoverability**: Is this truly unrecoverable? (Otherwise use WARN)
- [ ] **Brevity**: Is it ≤150 characters?
- [ ] **Correlation**: Is correlation_id propagated? (CRITICAL!)
- [ ] **Redaction**: Are secrets properly masked?

### FATAL Editorial Checklist
- [ ] **Severity**: Is this truly panic-worthy? (Otherwise use ERROR)
- [ ] **Complete context**: All state, inputs, and diagnostic info included?
- [ ] **Correlation**: Is correlation_id propagated?
- [ ] **Action taken**: What shutdown/abort action was triggered?
- [ ] **Redaction**: Even in crisis, secrets must be masked!

### DEBUG Editorial Checklist
- [ ] **Decision rationale**: Why did this happen?
- [ ] **Input context**: What were the inputs?
- [ ] **Intermediate state**: What's the current state?
- [ ] **Structured fields**: Using proper fields, not just strings?
- [ ] **Correlation**: Still tracking requests?
- [ ] **Purposeful**: Does this help debugging?

### TRACE Editorial Checklist
- [ ] **Exhaustive**: Every detail included?
- [ ] **Raw data**: Full payloads (redacted)?
- [ ] **Internal state**: Locks, queues, counters shown?
- [ ] **Entry/exit**: Function boundaries marked?
- [ ] **Still redacted**: Secrets still protected?

---

## 🔧 Configuration

### Environment Variables

```bash
# Set global narration level
export LLORCH_NARRATION_LEVEL=info    # mute|trace|debug|info|warn|error|fatal

# Set per-service level
export LLORCH_ORCHESTRATORD_NARRATION_LEVEL=debug
export LLORCH_POOL_MANAGERD_NARRATION_LEVEL=info
export LLORCH_WORKER_ORCD_NARRATION_LEVEL=trace

# Map to RUST_LOG for tracing integration
export RUST_LOG=info    # trace|debug|info|warn|error
```

### Runtime Configuration

```rust
// Set level programmatically
NarrationPolicy::set_level(NarrationLevel::Debug);

// Temporarily change level
let _guard = NarrationPolicy::with_level(NarrationLevel::Trace, || {
    // This block runs with TRACE level
    complex_operation().await?;
}); // Reverts to previous level
```

### Per-Module Configuration

```rust
// Enable DEBUG for specific module
narration_core::module_level("orchestratord::admission", NarrationLevel::Debug);

// Enable TRACE for specific function
narration_core::function_level("dispatch_job", NarrationLevel::Trace);
```

---

## 📊 Performance Impact

### Benchmarks

| Level | Events/sec | Overhead | Production Safe? |
|-------|-----------|----------|------------------|
| **MUTE** | 0 | 0% | ✅ Yes |
| **TRACE** | ~500,000 | ~25% | ❌ No |
| **DEBUG** | ~50,000 | ~5% | ⚠️ Incidents only |
| **INFO** | ~10,000 | <1% | ✅ Yes |
| **WARN** | ~10,000 | <1% | ✅ Yes |
| **ERROR** | ~10,000 | <1% | ✅ Yes |
| **FATAL** | ~1,000 | <1% | ✅ Yes (rare events) |

### Guidelines
- ✅ **INFO/WARN/ERROR/FATAL**: Always safe for production
- ⚠️ **DEBUG**: Use temporarily during incidents
- ❌ **TRACE**: Development/testing only
- 🔇 **MUTE**: Security/performance critical paths only

---

## 🧪 Testing Per Level

### INFO Tests
```rust
#[test]
fn test_info_level_events() {
    let capture = CaptureAdapter::install();
    NarrationPolicy::set_level(NarrationLevel::Info);
    
    // Should emit
    narrate_info(/* job accepted */);
    assert_eq!(capture.len(), 1);
    
    // Should NOT emit
    narrate_debug(/* internal detail */);
    assert_eq!(capture.len(), 1); // Still 1
}
```

### DEBUG Tests
```rust
#[test]
fn test_debug_level_events() {
    let capture = CaptureAdapter::install();
    NarrationPolicy::set_level(NarrationLevel::Debug);
    
    // Should emit INFO
    narrate_info(/* job accepted */);
    assert_eq!(capture.len(), 1);
    
    // Should emit DEBUG
    narrate_debug(/* decision point */);
    assert_eq!(capture.len(), 2);
    
    // Should NOT emit TRACE
    narrate_trace(/* loop iteration */);
    assert_eq!(capture.len(), 2); // Still 2
}
```

### TRACE Tests
```rust
#[test]
fn test_trace_level_events() {
    let capture = CaptureAdapter::install();
    NarrationPolicy::set_level(NarrationLevel::Trace);
    
    // Should emit everything
    narrate_info(/* job accepted */);
    narrate_debug(/* decision point */);
    narrate_trace(/* loop iteration */);
    assert_eq!(capture.len(), 3);
}
```

---

## 📚 Examples by Service

### orchestratord

**INFO**:
```rust
human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
human: "Dispatching job 'job-123' to worker-gpu0-r1 (ETA 420 ms)"
human: "Job completed in 2500 ms (150 tokens, 60 tokens/sec)"
```

**DEBUG**:
```rust
human: "Selecting pool: 'default' has 3 workers (1 busy, 2 idle)"
human: "Admission check: queue_depth=5, threshold=100, result=ACCEPT"
human: "Worker selection: worker-gpu0-r1 chosen (load=2/8, latency=50ms)"
```

**TRACE**:
```rust
human: "ENTER enqueue_job(job_id=job-123, pool_id=default)"
human: "Validating job.model_ref: value='llama-7b', result=OK"
human: "Acquiring admission_queue mutex (wait_time=1ms)"
human: "Queue depth transition: 5 → 6"
human: "EXIT enqueue_job → Result::Ok(position=6) (total_time=8ms)"
```

### pool-managerd

**INFO**:
```rust
human: "Worker ready with engine llamacpp-v1, 8 slots available"
human: "Spawning engine llamacpp-v1 for pool 'default' on GPU0"
human: "Pool 'default' transitioned from healthy to degraded (2/3 workers alive)"
```

**DEBUG**:
```rust
human: "Heartbeat received from worker-gpu0-r1 (last_seen=2500ms ago)"
human: "VRAM allocation: requested=2048MB, available=8192MB, result=OK"
human: "Engine spawn breakdown: download=500ms, load=1500ms, verify=100ms"
```

**TRACE**:
```rust
human: "ENTER register_worker(worker_id=worker-gpu0-r1, engine=llamacpp-v1)"
human: "Acquiring worker_registry mutex (wait_time=0ms)"
human: "Validating engine_version: value='v1.2.3', compatible=true"
human: "Updating worker state: status=live, slots=8, vram=8192MB"
human: "EXIT register_worker → Result::Ok(()) (total_time=5ms)"
```

### worker-orcd

**INFO**:
```rust
human: "Worker started successfully on GPU0 with engine llamacpp-v1"
human: "Processing inference request for model 'llama-7b' (150 tokens)"
human: "Inference completed in 2500 ms (150 tokens, 60 tokens/sec)"
```

**DEBUG**:
```rust
human: "Model loaded from cache: 'llama-7b' (load_time=50ms)"
human: "Tokenization: input=25 tokens, output=150 tokens (tokenize_time=10ms)"
human: "Inference breakdown: prefill=500ms, decode=2000ms (60 tokens/sec)"
```

**TRACE**:
```rust
human: "ENTER process_inference(model='llama-7b', max_tokens=150)"
human: "Tokenizing prompt: length=25 tokens"
human: "Loading model weights: shard 1/4 (512MB)"
human: "Loading model weights: shard 2/4 (512MB)"
human: "Prefill phase: processing 25 tokens (500ms)"
human: "Decode phase: token 1/150 generated (latency=13ms)"
human: "EXIT process_inference → Result::Ok(tokens=150) (total_time=2500ms)"
```

---

## 🎀 Cute Mode Per Level

### INFO + Cute
```rust
human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
cute: "Orchestratord welcomes job-123 to the queue! You're #3 in line! 🎫✨"
```

### DEBUG + Cute
```rust
human: "Selecting pool: 'default' has 3 workers (1 busy, 2 idle)"
cute: "Orchestratord looks around: 'default' pool has 3 friends, 2 are free! 👀"
```

### TRACE + Cute
```rust
human: "ENTER enqueue_job(job_id=job-123, pool_id=default)"
cute: "Orchestratord steps into enqueue_job with job-123 in hand! 🚪"
```

**Note**: Cute mode works at ALL levels! Even TRACE can be adorable! 🎀

---

## 🚨 Common Mistakes

### ❌ Wrong Level Selection

**MISTAKE**: Using INFO for internal details
```rust
// ❌ WRONG: This is DEBUG, not INFO
narrate_info(NarrationFields {
    human: "Loop iteration 47/100 processing worker-gpu2-r1".to_string(),
    ..Default::default()
});
```

**FIX**: Use DEBUG
```rust
// ✅ CORRECT: DEBUG for internal flow
narrate_debug(NarrationFields {
    human: "Processing worker 47/100: worker-gpu2-r1 (status=idle)".to_string(),
    ..Default::default()
});
```

### ❌ TRACE in Production

**MISTAKE**: Enabling TRACE in production
```bash
# ❌ WRONG: TRACE will overwhelm logs
export LLORCH_NARRATION_LEVEL=trace
```

**FIX**: Use INFO (or DEBUG for incidents)
```bash
# ✅ CORRECT: INFO for production
export LLORCH_NARRATION_LEVEL=info

# ✅ CORRECT: DEBUG for incident investigation
export LLORCH_NARRATION_LEVEL=debug
```

### ❌ MUTE for Convenience

**MISTAKE**: Using MUTE to hide noisy logs
```rust
// ❌ WRONG: Don't use MUTE to hide problems
NarrationPolicy::set_level(NarrationLevel::Mute);
noisy_function().await?;
```

**FIX**: Fix the noisy function or use appropriate level
```rust
// ✅ CORRECT: Fix the function to use proper levels
async fn noisy_function() {
    // Use DEBUG for internal details
    narrate_debug(/* ... */);
    
    // Use INFO for key events
    narrate_info(/* ... */);
}
```

---

## 📖 Summary

### Level Boundaries (TL;DR)

| Level | Boundary | Example |
|-------|----------|---------|
| **MUTE** | Complete silence | (nothing) |
| **TRACE** | Every single event | "Loop iteration 47/100" |
| **DEBUG** | Detailed flow + context | "Selected worker-gpu0-r1 (load=2/8)" |
| **INFO** | Key lifecycle events | "Job completed in 2500 ms" |
| **WARN** | Anomalies & degradations | "Retrying job-123 (attempt 2/5)" |
| **ERROR** | Operational failures | "VRAM allocation failed: 4GB requested, 2GB available" |
| **FATAL** | Unrecoverable errors | "CRITICAL: VRAM-only policy violated. Aborting." |

### When in Doubt

1. **Production?** → Use **INFO** (or **WARN**/**ERROR** for issues)
2. **Debugging?** → Use **DEBUG**
3. **Deep dive?** → Use **TRACE**
4. **Security?** → Use **MUTE** (sparingly)
5. **Anomaly?** → Use **WARN**
6. **Failure?** → Use **ERROR**
7. **Panic?** → Use **FATAL**

### Our Promise

With clear level boundaries, we will:
- ✅ Keep production logs clean (INFO)
- ✅ Surface actionable warnings (WARN)
- ✅ Provide rich error context (ERROR)
- ✅ Capture unrecoverable failures (FATAL)
- ✅ Provide rich debugging context (DEBUG)
- ✅ Enable exhaustive analysis (TRACE)
- ✅ Protect sensitive operations (MUTE)
- 🎀 Make every level adorable (cute mode!)

---

## 🚀 Implementation Notes

### Our Custom Proc Macros

We're building **custom proc macros** to make using these levels effortless:

```rust
// 95% of functions: just add #[trace_fn]!
#[trace_fn]  // Auto-infers actor from module path
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}

// User-facing events: use #[narrate(...)] with templates!
#[narrate(
    actor = "orchestratord",  // Or auto-inferred!
    action = "accept",
    human = "Accepted job {job_id} at position {position}",
    cute = "Orchestratord welcomes job-{job_id}! 🎫"
)]
fn accept_job(job_id: &str) -> Result<()> {
    // Our macro handles everything!
}
```

**See our complete documentation**:
- 📄 `FINAL_SUMMARY.md` — Complete 4-week plan
- 📄 `ERGONOMIC_TRACING.md` — Proc macro design
- 📄 `EXISTING_SOLUTIONS.md` — Why we built our own
- 📄 `DEVELOPER_EXPERIENCE.md` — Developer guidelines

---

*Reviewed by Narration Core Team — may your levels be clear, your actors be auto-inferred, and your narration be adorable! 🎀*
