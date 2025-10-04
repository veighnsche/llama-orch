# FT-049: Narration-Core Integration — Handoff from Narration-Core Team

**To**: worker-orcd Foundation-Alpha Team  
**From**: Narration-Core Team  
**Date**: 2025-10-04  
**Status**: 🎁 Ready for Integration

---

## What We Delivered

The narration-core team has completed our side of FT-049. Here's what we've prepared for you:

### 1. ✅ Worker-Specific Taxonomy

**Location**: `bin/shared-crates/narration-core/src/lib.rs`

We added constants for worker-orcd:

```rust
// Actors
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";

// Actions
pub const ACTION_INFERENCE_START: &str = "inference_start";
pub const ACTION_INFERENCE_COMPLETE: &str = "inference_complete";
pub const ACTION_HEARTBEAT_SEND: &str = "heartbeat_send";
pub const ACTION_READY_CALLBACK: &str = "ready_callback";
pub const ACTION_CANCEL: &str = "cancel";
```

### 2. ✅ Integration Guide

**Location**: `bin/shared-crates/narration-core/docs/WORKER_ORCD_INTEGRATION.md`

This is your **step-by-step guide** to integrate narration-core. It includes:

- **Step 1**: Add dependency to `Cargo.toml`
- **Step 2**: Import taxonomy constants
- **Step 3**: Extract correlation IDs from HTTP headers
- **Step 4**: Emit narration at critical paths
- **Step 5**: Propagate correlation IDs in outgoing requests
- **Step 6**: Follow editorial guidelines
- **Step 7**: Write tests
- **Step 8**: Verify integration
- **Step 9**: Submit for editorial review

**Complete code examples** for:
- Inference start
- Inference complete (with metrics)
- Heartbeat
- Ready callback
- Error handling

### 3. ✅ BDD Scenarios

**Location**: `bin/shared-crates/narration-core/bdd/features/worker_orcd_integration.feature`

We wrote **25 BDD scenarios** to verify your integration:

- Correlation ID propagation
- Performance metrics emission
- Editorial standards compliance
- Error context quality
- Secret redaction
- Distributed tracing support

These scenarios will help you verify your implementation is correct.

### 4. ✅ Editorial Guidelines

**Location**: Integration guide, section "Step 6: Editorial Guidelines"

We documented our editorial standards:

- ✅ **Clarity**: Can a developer understand what happened?
- ✅ **Specificity**: Are all relevant IDs/numbers included?
- ✅ **Brevity**: Is it under 100 characters?
- ✅ **Present tense**: "Starting inference" (not "Started")
- ✅ **Active voice**: "Worker sends heartbeat" (not "Heartbeat was sent")
- ✅ **Context**: Does it answer "why" not just "what"?
- ✅ **No secrets**: No bearer tokens, API keys, passwords
- ✅ **Correlation ID**: Included when available

---

## What You Need to Do

### Quick Start (30 minutes)

1. **Read the integration guide**: `bin/shared-crates/narration-core/docs/WORKER_ORCD_INTEGRATION.md`

2. **Add the dependency** to `bin/worker-orcd/Cargo.toml`:
   ```toml
   [dependencies]
   observability-narration-core = { path = "../shared-crates/narration-core" }
   ```

3. **Import the taxonomy** in your Rust files:
   ```rust
   use observability_narration_core::{
       narrate_auto,
       NarrationFields,
       ACTOR_WORKER_ORCD,
       ACTION_INFERENCE_START,
       ACTION_INFERENCE_COMPLETE,
   };
   ```

4. **Extract correlation IDs** from HTTP headers:
   ```rust
   fn extract_correlation_id(headers: &HeaderMap) -> Option<String> {
       headers
           .get("X-Correlation-Id")
           .and_then(|v| v.to_str().ok())
           .map(String::from)
   }
   ```

5. **Emit narration** at critical paths (see integration guide for examples)

### Critical Paths to Narrate

You **MUST** emit narration at these points:

1. **Inference start** — When job begins
2. **Inference complete** — When tokens generated (include duration_ms, tokens_out)
3. **Heartbeat** — Every heartbeat to pool-managerd
4. **Ready callback** — When worker becomes live
5. **Errors** — All error paths with specific context

### Example: Inference Start

```rust
narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_INFERENCE_START,
    target: job_id.clone(),
    correlation_id: Some(correlation_id.clone()),
    model_ref: Some("llama-7b".to_string()),
    tokens_in: Some(prompt_tokens),
    human: format!("Starting inference for job {} with model llama-7b", job_id),
    ..Default::default()
});
```

### Example: Inference Complete

```rust
let elapsed = start_time.elapsed();

narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_INFERENCE_COMPLETE,
    target: job_id.clone(),
    correlation_id: Some(correlation_id.clone()),
    duration_ms: Some(elapsed.as_millis() as u64),
    tokens_out: Some(generated_tokens),
    human: format!("Completed inference: {} tokens in {} ms", generated_tokens, elapsed.as_millis()),
    ..Default::default()
});
```

---

## Acceptance Criteria (Your Checklist)

- [ ] `observability-narration-core` dependency added to `Cargo.toml`
- [ ] Taxonomy constants imported
- [ ] Correlation IDs extracted from HTTP headers (`X-Correlation-Id`)
- [ ] Inference start narration emitted
- [ ] Inference complete narration emitted (with duration_ms, tokens_out)
- [ ] Heartbeat narration emitted
- [ ] Ready callback narration emitted
- [ ] Error narrations include specific context (not generic "error occurred")
- [ ] Correlation IDs propagated in outgoing HTTP requests
- [ ] Unit tests written using `CaptureAdapter`
- [ ] Verification commands run successfully
- [ ] Submitted for editorial review

---

## Verification Commands

After integration, run these to verify:

```bash
# Build worker-orcd
cargo build -p worker-orcd

# Run worker and capture logs
RUST_LOG=info cargo run -p worker-orcd 2>&1 | tee worker.log

# Check for narration events
grep "actor=worker-orcd" worker.log

# Check for correlation IDs
grep "correlation_id=" worker.log

# Check for secrets (should be redacted)
grep -i "bearer\|api_key\|password" worker.log
```

---

## Editorial Review Process

Once you've integrated, **submit for editorial review**:

1. **Notify us**: Let narration-core team know you're ready
2. **Provide sample logs**: Run worker-orcd and share logs
3. **We'll review**: We'll check narration quality against our editorial standards
4. **We'll provide feedback**: Suggestions for improvement (if needed)
5. **Approval**: Once approved, FT-049 is complete! 🎉

---

## Support

We're here to help! If you have questions:

- **Read the integration guide**: `docs/WORKER_ORCD_INTEGRATION.md`
- **Check our README**: `bin/shared-crates/narration-core/README.md`
- **Review our team doc**: `TEAM_RESPONSIBILITY.md`
- **Ask us directly**: We're the narration-core team! 🎀

---

## Why This Matters

Narration-core enables **distributed tracing** across llama-orch:

```
orchestratord → worker-orcd → pool-managerd
     ↓               ↓               ↓
correlation_id  correlation_id  correlation_id
```

With proper integration, you'll be able to:

1. **Trace requests** across services via correlation IDs
2. **Debug failures** with human-readable stories
3. **Measure performance** with structured metrics
4. **Monitor production** with observability tools

This is **foundational infrastructure** for the entire project. Thank you for integrating! 💕

---

## Timeline

- **Day 73** (Today): Narration-core deliverables complete ✅
- **Day 74** (Tomorrow): Worker-orcd integration complete ⏳
- **Day 74** (EOD): Editorial review and approval ⏳

---

## Questions?

We're the **cutest, most helpful observability team** in the monorepo. We're here to make your debugging experience delightful! 🎀

Just reach out. We'll help. Promise. 💝

---

**Status**: 🎁 Ready for Integration  
**Next Action**: Worker-orcd Foundation-Alpha begins implementation  
**Blocking**: None (all narration-core deliverables complete)

---

*Handoff prepared by the Narration Core Team — may your correlation IDs be present and your logs be readable! 🎀*
