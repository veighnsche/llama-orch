# FT-049: Worker-Orcd Narration Integration (Narration-Core Perspective)

**Team**: Narration-Core  
**Consumer**: worker-orcd (Foundation-Alpha)  
**Sprint**: Sprint 7 - Final Integration  
**Size**: S (1 day)  
**Days**: 73  
**Spec Ref**: TEAM_RESPONSIBILITY.md (Ultimate Editorial Authority)

---

## What the PM Missed

The PM wrote FT-049 as if Foundation-Alpha needs to "integrate narration-core logging patterns" internally. But **we ARE narration-core**. The real story is:

> **worker-orcd needs to integrate with us, and we need to support them.**

---

## Our Responsibilities

As narration-core, we need to:

1. **Provide worker-specific taxonomy** (actors, actions)
2. **Write integration guide** for worker-orcd
3. **Create BDD scenarios** to verify correct usage
4. **Review their narration** (we have ultimate editorial authority!)
5. **Ensure correlation ID propagation** from orchestratord → worker-orcd

---

## Worker-Orcd Integration Checklist

### 1. Taxonomy Extensions

Add to `src/lib.rs`:

```rust
// Worker-specific actors
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";

// Worker-specific actions
pub const ACTION_INFERENCE_START: &str = "inference_start";
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";
pub const ACTION_HEARTBEAT_SEND: &str = "heartbeat_send";
pub const ACTION_READY_CALLBACK: &str = "ready_callback";
pub const ACTION_CANCEL: &str = "cancel";
```

### 2. Integration Guide

Create `docs/WORKER_ORCD_INTEGRATION.md`:

```markdown
# Worker-Orcd Integration Guide

## Add Dependency

\`\`\`toml
[dependencies]
observability-narration-core = { path = "../shared-crates/narration-core" }
\`\`\`

## Basic Usage

\`\`\`rust
use observability_narration_core::{narrate_auto, ACTOR_WORKER_ORCD, ACTION_INFERENCE_START};

narrate_auto!(
    actor = ACTOR_WORKER_ORCD,
    action = ACTION_INFERENCE_START,
    target = job_id,
    correlation_id = req.correlation_id,
    model_ref = "llama-7b",
    human = "Starting inference for job {job_id}"
);
\`\`\`

## Correlation ID Propagation

Extract from HTTP headers:

\`\`\`rust
let correlation_id = req.headers()
    .get("X-Correlation-Id")
    .and_then(|v| v.to_str().ok())
    .unwrap_or("unknown");
\`\`\`

## Critical Path Logs

- **Inference start**: When job begins
- **Inference complete**: When tokens generated
- **Heartbeat**: Every heartbeat to pool-managerd
- **Ready callback**: When worker becomes live
- **Errors**: All error paths with context
```

### 3. BDD Scenarios

Create `bdd/features/worker_orcd_integration.feature`:

```gherkin
Feature: Worker-Orcd Narration Integration

  Scenario: Worker emits inference start narration
    Given worker-orcd receives inference request with correlation_id "req-123"
    When worker starts inference
    Then narration event is emitted with:
      | field          | value              |
      | actor          | worker-orcd        |
      | action         | inference_start    |
      | correlation_id | req-123            |
      | human          | Starting inference |

  Scenario: Worker propagates correlation IDs
    Given orchestratord sends request with correlation_id "req-abc"
    When worker-orcd processes the request
    Then all narration events include correlation_id "req-abc"

  Scenario: Worker emits performance metrics
    Given worker completes inference in 2500 ms
    When worker emits completion narration
    Then narration includes:
      | field       | value |
      | duration_ms | 2500  |
      | tokens_out  | 150   |
```

### 4. Editorial Review

We will review worker-orcd's narration for:

- ✅ **Correlation ID discipline**: Every event has correlation_id
- ✅ **Human-readable**: Clear, specific, under 100 chars
- ✅ **Performance metrics**: duration_ms, tokens_out on completion
- ✅ **Error context**: Specific error messages, not generic "error occurred"
- ✅ **Secret redaction**: No bearer tokens in logs

---

## Acceptance Criteria (Narration-Core)

- [ ] Worker-specific actors/actions added to taxonomy
- [ ] Integration guide written
- [ ] BDD scenarios created
- [ ] Editorial review checklist defined
- [ ] Example narrations provided
- [ ] Correlation ID propagation documented

---

## Acceptance Criteria (Worker-Orcd)

*These are worker-orcd's responsibilities, but we'll verify:*

- [ ] `observability-narration-core` dependency added
- [ ] All critical paths emit narration
- [ ] Correlation IDs extracted from HTTP headers
- [ ] Correlation IDs included in all narration
- [ ] Performance metrics logged (duration, tokens)
- [ ] Error narrations include context
- [ ] No secrets in logs

---

## Example Narrations for Worker-Orcd

### Inference Start
```rust
narrate_auto!(
    actor = ACTOR_WORKER_ORCD,
    action = ACTION_INFERENCE_START,
    target = job_id,
    correlation_id = correlation_id,
    model_ref = "llama-7b",
    human = "Starting inference for job {job_id} with model llama-7b"
);
```

### Inference Complete
```rust
narrate_auto!(
    actor = ACTOR_WORKER_ORCD,
    action = ACTION_INFERENCE_COMPLETE,
    target = job_id,
    correlation_id = correlation_id,
    duration_ms = elapsed.as_millis(),
    tokens_out = token_count,
    human = "Completed inference: {token_count} tokens in {duration_ms} ms"
);
```

### Heartbeat
```rust
narrate_auto!(
    actor = ACTOR_WORKER_ORCD,
    action = ACTION_HEARTBEAT_SEND,
    target = "pool-managerd",
    correlation_id = correlation_id,
    human = "Sending heartbeat to pool-managerd"
);
```

### Error
```rust
narrate_auto!(
    actor = ACTOR_WORKER_ORCD,
    action = ACTION_INFERENCE_START,
    target = job_id,
    correlation_id = correlation_id,
    error_kind = "cuda_oom",
    human = "Inference failed: CUDA out of memory (requested 4GB, only 2GB available)"
);
```

---

## Dependencies

**Upstream**: None (we're foundational!)  
**Downstream**: worker-orcd Foundation-Alpha team

---

## Definition of Done

- [ ] Taxonomy extended
- [ ] Integration guide written
- [ ] BDD scenarios created
- [ ] Editorial checklist published
- [ ] worker-orcd team notified
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Narration-Core  
**Created**: 2025-10-04

---

*Prepared by the Narration Core Team — may your correlation IDs be present and your logs be readable! 🎀*
